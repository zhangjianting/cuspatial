/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <time.h>
#include <sys/time.h>
#include <string>
#include <random>
#include <algorithm>
#include <functional>

#include <gtest/gtest.h>
#include <utilities/legacy/error_utils.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>

#include <ogrsf_frmts.h>
#include <geos_c.h>


#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>

#include <utility/utility.hpp>
#include <utility/quadtree_thrust.cuh>
#include <utility/bbox_thrust.cuh>

#include <cuspatial/quadtree.hpp>
#include <cuspatial/polygon_bbox.hpp>
#include <cuspatial/spatial_jion.hpp>

#include "spatial_join_test_utility.hpp"

/*
* Test code for running spatial join on GPUs with GDAL/OGR-based CPU verification.
* Different from spatial_join_refinement_small and spatial_join_refinement_large,
* where the expected results can be either embedded in code or computed on CPU (using GDAL/OGR API),
* for yearly NYC taxitrip data, that would take donzes of hours or even more. 
* A sampling-based verificaiton is thus needed. 
*
* Two sampling strategies are provided: sampling on points and sampling on quadrant-polygon pairs.
*
* Furthermore, sophisticated polygons such as multi-polgyons and polygons with holes, 
* are challenging for verfication/debugging purposes .
* This test code allows picking up two types of polygons, i.e., single-ring (classical) 
* and multi-ring polygons and their combinations (all). 
*
* The comparison/vericiation code is also accelerated by using a composition of 
* lower_bound/upper_bound/binary_search paralell primiitves in Thrust. 
* Searching std::vector on CPU is just too slow for this prurpose.   

* As the relationship between points and polygons is many-to-many, the verification gives three metrics: 
* num_search_pnt: numbers of points (from both sampling strategies) that are within at least
* one polygons by CPU code; Disagreement between num_search_pnt and num_pp_pairs indciate mismatches;
* num_not_found: # of point indices of GDAL/OGR CPU results can not be found in GPU results
* num_mis_match: for the same point index,if its assoicated non-empty polygon sets are different 
* between CPU and GPU results, num_mis_match will be increased by 1.

* For a perfert agreement between CPU and GPU results, 
* (num_search_pnt==num_pp_pairs && num_not_found==0 && num_mis_match==0);
*
* naming convention: *_pnt_* (points), *_poly_* (polygon), *_quad_* (quadrant) 
* naming convention: *_qt_* (quadtree), *_pq_* (polygon-quadrant pair), *_pp_* (polygon-point pair) 
* naming convention: h_*(host vairable), d_*(device variable), *_vec (std::vector), *_idx_ (index/offset)
*/

struct SpatialJoinNYCTaxiVerify : public GdfTest 
{        
    uint32_t num_pnts=0;

    uint32_t num_quadrants=0;

    uint32_t num_pq_pairs=0;

    uint32_t num_pp_pairs=0;
   
     //point x/y on host
    double *h_pnt_x=nullptr,*h_pnt_y=nullptr;

    uint32_t num_poly=0,num_ring=0,num_vertex=0;

    //polygon vertices x/y
    double *h_poly_x=nullptr,*h_poly_y=nullptr;

    //quadtree length/fpos
    uint32_t *h_qt_length=nullptr,*h_qt_fpos=nullptr;   

    //quadrant/polygon pairs
    uint32_t *h_pq_quad_idx=nullptr,*h_pq_poly_idx=nullptr;   
    
    uint32_t *h_pp_pnt_idx=nullptr,*h_pp_poly_idx=nullptr;

    //poygons using GDAL/OGR OGRGeometry structure
    std::vector<OGRGeometry *> h_ogr_polygon_vec;
    std::vector<GEOSGeometry *> h_geos_polygon_vec;

    //sequential idx 0..num_poly-1 to index h_ogr_polygon_vec
    //needed when actual polygons in spatial join are only a subset, e.g., multi-polygons only  
    std::vector<uint32_t> h_org_poly_idx_vec;

    //point idx that intersect with at least one polygon based on GDAL/OGR OGRGeometry.Contains 
    std::vector<uint32_t> h_pnt_idx_vec;
    
    //# of poylgons that are contain points indexed by h_pnt_idx_vec at the same index
    std::vector<uint32_t> h_pnt_len_vec;

    //#polygon indices for those contain points in h_pnt_idx_vec; sequentially concatenated
    std::vector<uint32_t> h_poly_idx_vec;


    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();

    SBBox<double> setup_polygons(const char *file_name,uint8_t type)
    {
        std::vector<int> g_len_v,f_len_v,r_len_v;
        std::vector<double> x_v, y_v;
        GDALAllRegister();
        GDALDatasetH hDS = GDALOpenEx(file_name, GDAL_OF_VECTOR, nullptr, nullptr, nullptr );
        if(hDS==nullptr)
        {
            std::cout<<"Failed to open ESRI Shapefile dataset "<< file_name<<std::endl;
            exit(-1);
        }
        //a shapefile abstracted as a GDALDatasetGetLayer typically has only one layer
        OGRLayerH hLayer = GDALDatasetGetLayer( hDS,0 );

        this->h_ogr_polygon_vec.clear();
        this->h_geos_polygon_vec.clear();
        this->h_org_poly_idx_vec.clear();
        
        //type: 0 for all, 1 for simple polygons and 2 for multi-polygons
        uint32_t num_f=ReadLayer(hLayer,g_len_v,f_len_v,r_len_v,x_v,y_v,type,h_ogr_polygon_vec,h_org_poly_idx_vec);
        assert(num_f>0);
        
        h_geos_polygon_vec.clear();
        GEOSContextHandle_t hGEOSCtxt = OGRGeometry::createGEOSContext();
        for(uint32_t i=0;i<num_f;i++)
        {
            OGRGeometry *poOGRPoly=h_ogr_polygon_vec[i];
            GEOSGeometry *poGEOSPoly = poOGRPoly->exportToGEOS(hGEOSCtxt);
            h_geos_polygon_vec.push_back(poGEOSPoly);      	
        }

        //num_group=g_len_v.size();
        this->num_poly=f_len_v.size();
        this->num_ring=r_len_v.size();
        this->num_vertex=x_v.size();

        uint32_t *h_poly_flen=new uint32_t[num_poly];
        uint32_t *h_poly_rlen=new uint32_t[num_ring];
        assert(h_poly_flen!=nullptr && h_poly_rlen!=nullptr);
        
        this->h_poly_x=new double [num_vertex];
        this->h_poly_y=new double [num_vertex];
        assert(h_poly_x!=nullptr && h_poly_y!=nullptr);

        std::copy_n(f_len_v.begin(),num_poly,h_poly_flen);
        std::copy_n(r_len_v.begin(),num_ring,h_poly_rlen);
        std::copy_n(x_v.begin(),num_vertex,h_poly_x);
        std::copy_n(y_v.begin(),num_vertex,h_poly_y);
        std::cout<<"setup_polygons: num_poly="<<num_poly<<" num_ring="<<num_ring<<" num_vertex="<<num_vertex<<std::endl;

        //note that the bbox of all polygons will used as the Area of Intersects (AOI) to join points with polygons 
        double x1=*(std::min_element(x_v.begin(),x_v.end()));
        double x2=*(std::max_element(x_v.begin(),x_v.end()));
        double y1=*(std::min_element(y_v.begin(),y_v.end()));
        double y2=*(std::max_element(y_v.begin(),y_v.end()));
        std::cout<<"read_polygon_bbox: x_min="<<x1<<"  y_min="<<y1<<" x_max="<<x2<<" y_max="<<y2<<std::endl;

        return SBBox<double>(thrust::make_tuple(x1,y1), thrust::make_tuple(x2,y2));
    }

    void compare_random_points(uint32_t num_samples,uint32_t num_print_interval,bool using_geos)
    {
        std::cout<<"compare_random_points: num_quadrants="<<this->num_quadrants
            <<" num_pp_pair="<<this->num_pp_pairs<<" num_samples="<<num_samples<<std::endl;
        
        std::vector<uint32_t> rand_indices;
        gen_rand_idx(rand_indices,this->num_pnts,num_samples);

        timeval t0,t1;
        gettimeofday(&t0, nullptr);

        //h_pnt_idx_vec, h_pnt_len_vec and h_poly_idx_vec will be cleared first
  
        if(using_geos)
        {
            rand_points_geos_pip_test(num_print_interval,rand_indices, this->h_geos_polygon_vec,this->h_pnt_idx_vec,
                this->h_pnt_len_vec,this->h_poly_idx_vec,this->h_pnt_x,this->h_pnt_y);
        }
        else
        {
            rand_points_ogr_pip_test(num_print_interval,rand_indices, this->h_ogr_polygon_vec,this->h_pnt_idx_vec,
                this->h_pnt_len_vec,this->h_poly_idx_vec,this->h_pnt_x,this->h_pnt_y);
         }       
        gettimeofday(&t1, nullptr);
        float cpu_time=cuspatial::calc_time("cpu random sampling computing time = ",t0,t1);
    }
  
    void compare_matched_pairs(uint32_t num_samples,uint32_t num_print_interval,bool using_geos)
    {
        std::cout<<"compare_matched_pairs: num_quadrants="<<this->num_quadrants<<" num_pq_pairs"<<this->num_pq_pairs
            <<" num_pp_pair="<<this->num_pp_pairs<<" num_samples="<<num_samples<<std::endl;

        std::vector<uint32_t> rand_indices;
        gen_rand_idx(rand_indices,this->num_pq_pairs,num_samples);

        timeval t0,t1;
        gettimeofday(&t0, nullptr);
        
        if(using_geos)
        {
            matched_pairs_geos_pip_test(num_print_interval,rand_indices,
                this->h_pq_quad_idx,this->h_pq_poly_idx,this->h_qt_length,this->h_qt_fpos,
                this->h_geos_polygon_vec,this->h_pnt_idx_vec,this->h_pnt_len_vec,this->h_poly_idx_vec,
                this->h_pnt_x,this->h_pnt_y);
        }
        else
        {
            matched_pairs_ogr_pip_test(num_print_interval,rand_indices,
                this->h_pq_quad_idx,this->h_pq_poly_idx,this->h_qt_length,this->h_qt_fpos,
                this->h_ogr_polygon_vec,this->h_pnt_idx_vec,this->h_pnt_len_vec,this->h_poly_idx_vec,
                this->h_pnt_x,this->h_pnt_y);
   
        }
        gettimeofday(&t1, nullptr);
        float cpu_time=cuspatial::calc_time("cpu matched-pair computing time",t0,t1);                
    }

    void read_nyc_taxi(const char *file_name)
    {
        CUDF_EXPECTS(file_name!=NULL,"file_name can not be NULL");
        FILE *fp=fopen(file_name,"rb");
        CUDF_EXPECTS(fp!=NULL, "can not open file for input");
        CUDF_EXPECTS(fread(&(this->num_pnts),sizeof(uint32_t),1,fp)==1,"reading num_pnt failed");
        CUDF_EXPECTS(fread(&(this->num_quadrants),sizeof(uint32_t),1,fp)==1,"reading num_quadrants failed");
        CUDF_EXPECTS(fread(&(this->num_pq_pairs),sizeof(uint32_t),1,fp)==1,"reading num_pq_pairs failed");
        CUDF_EXPECTS(fread(&(this->num_pp_pairs),sizeof(uint32_t),1,fp)==1,"reading num_pp_pairs failed");
        std::cout<<"num_pnts="<<num_pnts<<" num_quadrants="<<num_quadrants<<" num_pq_pairs="<<num_pq_pairs<<" num_pp_pairs="<<num_pp_pairs<<std::endl;
    
        std::cout<<"reading points..."<<std::endl;
        this->h_pnt_x=new double[this->num_pnts];
        this->h_pnt_y=new double[this->num_pnts];
        CUDF_EXPECTS( this->h_pnt_x!=NULL && this->h_pnt_y!=NULL,"allocating memory for points on host failed");
    
        CUDF_EXPECTS(fread(this->h_pnt_x,sizeof(double),this->num_pnts,fp)==this->num_pnts,"reading h_pnt_x failed");
        CUDF_EXPECTS(fread(this->h_pnt_y,sizeof(double),this->num_pnts,fp)==this->num_pnts,"reading h_pnt_y failed");
        
        std::cout<<"reading quadrants..."<<std::endl;
        this->h_qt_length=new uint32_t[this->num_quadrants];
        this->h_qt_fpos=new uint32_t[this->num_quadrants];
        CUDF_EXPECTS( this->h_qt_length!=NULL && this->h_qt_fpos!=NULL,"allocating memory for quadrants on host failed");
  
        CUDF_EXPECTS(fread(this->h_qt_length,sizeof(uint32_t),this->num_quadrants,fp)==this->num_quadrants,"reading h_qt_length failed");
        CUDF_EXPECTS(fread(this->h_qt_fpos,sizeof(uint32_t),this->num_quadrants,fp)==this->num_quadrants,"reading h_qt_fpos failed");

        std::cout<<"reading quadrant/polygon pairs..."<<std::endl;
        this->h_pq_quad_idx=new uint32_t[this->num_pq_pairs];    
        this->h_pq_poly_idx=new uint32_t[this->num_pq_pairs];
        CUDF_EXPECTS( this->h_pq_poly_idx!=NULL && this->h_pq_quad_idx!=NULL,"allocating memory for quadrant-polygon pairs on host failed");

        CUDF_EXPECTS(fread(this->h_pq_quad_idx,sizeof(uint32_t),this->num_pq_pairs,fp)==this->num_pq_pairs,"reading h_pq_quad_idx failed");
        CUDF_EXPECTS(fread(this->h_pq_poly_idx,sizeof(uint32_t),this->num_pq_pairs,fp)==this->num_pq_pairs,"reading h_pq_poly_idx failed");

        std::cout<<"reading point/polygon pairs..."<<std::endl;
        this->h_pp_poly_idx=new uint32_t[this->num_pp_pairs];
        this->h_pp_pnt_idx=new uint32_t[this->num_pp_pairs];    
        CUDF_EXPECTS(this->h_pp_poly_idx!=NULL && this->h_pp_pnt_idx!=NULL,"allocating memory for point-polygon pairs on host failed");

        CUDF_EXPECTS(fread(this->h_pp_poly_idx,sizeof(uint32_t),this->num_pp_pairs,fp)==this->num_pp_pairs,"reading h_pp_poly_idx failed");
        CUDF_EXPECTS(fread(this->h_pp_pnt_idx,sizeof(uint32_t),this->num_pp_pairs,fp)==this->num_pp_pairs,"reading h_pp_pnt_idx failed");   

if(0)
{
        for(uint32_t i=0;i<this->num_pp_pairs;i++)
        {
            if(i%100==0)
                std::cout<<i<<" "<<h_pp_poly_idx[i]<<" "<<h_pp_pnt_idx[i]<<std::endl;
        }
}

    }

    void tear_down()
    {
        delete[] this->h_poly_x; this->h_poly_x=nullptr;
        delete[] this->h_poly_y; this->h_poly_y=nullptr;

        delete[] this->h_pnt_x; this->h_pnt_x=nullptr;
        delete[] h_pnt_y; h_pnt_y=nullptr;
        
        delete[] this->h_pq_quad_idx; this->h_pq_quad_idx=nullptr;
        delete[] h_pq_poly_idx; h_pq_poly_idx=nullptr;
        
        delete[] this->h_qt_length; this->h_qt_length=nullptr;
        delete[] this->h_qt_fpos; this->h_qt_fpos=nullptr;
    }

};

/* 
 * There could be multple configureations (minior ones are inside parentheses): 
 * pick one of three polygon datasets
 * choose from compare_random_points and compare_matched_pairs 
*/

TEST_F(SpatialJoinNYCTaxiVerify, verify)
{
    const char* env_p = std::getenv("CUSPATIAL_DATA");
    CUDF_EXPECTS(env_p!=nullptr,"CUSPATIAL_DATA environmental variable must be set");
   
    //#0: NYC taxi zone: 263 polygons
    //from https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip
    //#1: NYC Community Districts: 71 polygons
    //from https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nycd_11aav.zip
    //#2: NYC Census Tract 2000 data: 2216 polygons
    //from: https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyct2000_11aav.zip
 
    //note that the polygons and the points need to use the same projection 
    //all the three polygon datasets use epsg:2263 (unit is foot) for NYC/Long Island area 

    enum POLYID {taxizone_id=0,cd_id,ct_id};    
    POLYID sel_id=taxizone_id;

    const char * shape_files[]={"taxi_zones.shp","nycd_11a_av/nycd.shp","nyct2000_11a_av/nyct2000.shp"};
    
    const char * bin_files[]={"nyc_taxizone_2009_1.bin","nyc_cd_2009_12.bin","nyc_ct_2009_12.bin"};
    
    read_nyc_taxi(bin_files[sel_id]);

    std::cout<<"loading NYC polygon data..........."<<std::endl;

    std::string shape_filename=std::string(env_p)+std::string(shape_files[sel_id]); 
    
    std::cout<<"Using shapefile "<<shape_filename<<std::endl;

    //uint8_t poly_type=2; //multi-polygons only 
    //uint8_t poly_type=1; //single-polygons only 
    uint8_t poly_type=0; //all polygons

    this->setup_polygons(shape_filename.c_str(),poly_type);

    std::cout<<"running GDAL/OGR or GEOS CPU code for comparison/verification..........."<<std::endl;

    uint32_t num_print_interval=100;
    
    bool using_geos=false;

    //type 1: random points
    //uint32_t num_pnt_samples=this->num_pnts;
    uint32_t num_pnt_samples=10000;
    this->compare_random_points(num_pnt_samples,num_print_interval,using_geos);

    //type 2: random quadrant/polygon pairs
    //uint32_t num_quad_samples=10000;
    //this->compare_matched_pairs(num_quad_samples,num_print_interval,using_geos);

    //for unknown reason, the following two lines can not be compiled in spatial_join_test_utility.cu
    //h_pnt_search_idx and h_poly_search_idx do not need to be freed as the destructor of std::vector does it
    uint32_t * h_pnt_search_idx=&(h_pnt_idx_vec[0]);
    uint32_t * h_poly_search_idx=&(h_poly_idx_vec[0]);

    bool verified=compute_mismatch(this->num_pp_pairs,this->h_org_poly_idx_vec,
        h_pnt_search_idx,this->h_pnt_len_vec,h_poly_search_idx,
        this->h_pp_pnt_idx,this->h_pp_poly_idx,   
        this->h_pnt_x,this->h_pnt_y,mr,stream);
    std::string msg=verified ? "verified" : "mismatch";
    std::cout<<"comparison/verification result: " << msg << std::endl;
    this->tear_down();

}//TEST_F

