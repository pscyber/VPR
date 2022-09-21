/**
* This file is part of ibow-lcd.
*
* Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* ibow-lcd is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ibow-lcd is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ibow-lcd. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <ctime>
#include <deque> 
#include <boost/filesystem.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>     
#include <sys/time.h>
int score=0;
int id1=0,id2=0;
using namespace cv;
using namespace std;

struct node{
  vector<Point> contours;  
  Vec4i hierarchy;
  Point p;
  int cls=9999;
  vector<int> twoDnbID;
  vector<float> twoDnbSize;
  vector<Point> twoDnbCenters;
  float size=0;
};

struct tuceng{

  int leibie=9999;
  int contourNum=0;
  vector<Point> Centers;
  float size=0;
};

vector<vector<int>> classes;
vector<vector<node>> allimagenodes;
vector<vector<tuceng>> allimagetucengs;
int huihuancishu=0;
int lastCLflag=0;
int lastCLid=0;
deque<int> l_5_id;
deque<int> l_5_flag;
int w=0;
int h=0;
float size_img=0;
ofstream file("/home/serve/Desktop/synthia/9.24/xxx.csv");
struct timeval tv1,tv2;
long long T;
double skymianji=0;
double skyroad=0;
double skysidewalk=0;
double skybuild=0;
double skyfence=0;
double skypole=0;
double skytrafficsign=0;
double skyvegetation=0;
double skytrafficlight=0;
double skyroadline=0;

double skyn=0;
double roadn=0;
double sidewalkn=0;
double buildn=0;
double fencen=0;
double polen=0;
double trafficsignn=0;
double vegetationn=0;
double trafficlightn=0;
double roadlinen=0;
long long pjtime=0;
long long pjtime1=0;
void getFilenames(const std::string& directory,
                  std::vector<std::string>* filenames) {
    using namespace boost::filesystem;

    filenames->clear();
    path dir(directory);

    // Retrieving, sorting and filtering filenames.
    std::vector<path> entries;
    copy(directory_iterator(dir), directory_iterator(), back_inserter(entries));
    sort(entries.begin(), entries.end());
    for (auto it = entries.begin(); it != entries.end(); it++) {
        std::string ext = it->extension().c_str();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".png" || ext == ".jpg" ||
            ext == ".ppm" || ext == ".jpeg") {
            filenames->push_back(it->string());
        }
    }
}

void calculateNode(cv::Mat img1)
{
     w=img1.cols;
     h=img1.rows;
     size_img=w*h;
       // Processing image i
    //std::cout << "--- Processing image " << i << std::endl;
    vector<node> imagenodes;
    vector<tuceng> imagetucengs;
    vector<vector<Point>> contoursSky;  
    vector<Vec4i> hierarchySky;
    // Loading and describing the image
   // cv::Mat img1 = cv::imread(filenames[1]);
    cv::Mat img2;
    img2.create(img1.size(),img1.type());
    Mat imageContours=Mat::zeros(img1.size(),CV_8UC1);  
    Mat Contours=Mat::zeros(img1.size(),CV_8UC1);  //绘制 
    //cout<<img1.size()<<endl;

    for(int c_size=0;c_size<classes.size();c_size++)
    {
     int R=classes[c_size][0];
     int G=classes[c_size][1];
     int B=classes[c_size][2];
    for(int i=0;i<img1.rows;i++)
    {
      for(int j=0;j<img1.cols;j++)
      {
        int b,g,r;
        r=img1.at<cv::Vec3b>(i,j)[2];
        g=img1.at<cv::Vec3b>(i,j)[1];
        b=img1.at<cv::Vec3b>(i,j)[0];
        
        
        
        //cout<<r<<" "<<g<<" "<<b<<endl;
      
      if(i<2 || i>(img1.rows-2) || j<2 || j>(img1.cols-2))
      {
        img2.at<cv::Vec3b>(i,j)[0]=255;
        img2.at<cv::Vec3b>(i,j)[1]=255;
        img2.at<cv::Vec3b>(i,j)[2]=255; 
      }
      else{
        if(r==R && g==G && b==B)
      {
        //cout<<"car***************"<<endl;
        img2.at<cv::Vec3b>(i,j)[0]=0;
        img2.at<cv::Vec3b>(i,j)[1]=0;
        img2.at<cv::Vec3b>(i,j)[2]=0; 
      }
        else{
        img2.at<cv::Vec3b>(i,j)[0]=255;
        img2.at<cv::Vec3b>(i,j)[1]=255;
        img2.at<cv::Vec3b>(i,j)[2]=255;  
        }
      }  
      }
    }
    //闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象
    Mat dilated_cv;
    dilate(img2, dilated_cv, Mat());
    Mat eroded_cv;
    erode(dilated_cv, eroded_cv, Mat());
    vector<vector<Point>> contours;  
    vector<Vec4i> hierarchy;
    //Mat gray1;
   // copyMakeBorder(eroded_cv, gray1, 20, 20, 20, 20, BORDER_CONSTANT,Scalar::all(255)); 
    Mat gray2;
    cvtColor(dilated_cv, gray2,COLOR_BGR2GRAY);  //转换到GRAY空间（灰度图）
    //Mat gray;
    //GaussianBlur(gray2,gray,Size(5,5),0,0);
    //floodFill(gray2,Point(0,0),Scalar(255));
   
    //imshow("erzhihua",gray);
    //waitKey(2000);
    
    //cout<<gray.size()<<endl;
    //xunzhaobianjie
    findContours(gray2,contours,hierarchy,CV_RETR_LIST ,CV_CHAIN_APPROX_NONE,Point());
    
    
    //cout<<contours.size()<<endl;
    vector<vector<Point>> contours1;  
    vector<Vec4i> hierarchy1;
     int a=0;
 
    switch (c_size)
     {
     case 0: a=0.0008*size_img; break;
     case 1: a=0.0008*size_img; break;
     case 2: a=0.0008*size_img; break;
     case 3: a=0.0008*size_img; break;
     case 4: a=0.00160*size_img; break;
     case 5: a=0.0002292*size_img; break;
     case 6: a=30; break;
     case 7: a=30; break;
     case 8: a=0.00160*size_img; break;
     case 9: a=30; break;
     case 10: a=0.0006729*size_img; break;
     case 11: a=0.0002292*size_img; break;

     default:
       break;
     }
     //duilunkuojinxingguolv,guolvtukuang,gulvxiaolunkuo
    for(int b=0;b<contours.size();b++) {
           //绘制出contours向量内所有的像素点  
            Point P=Point(contours[b][0].x,contours[b][0].y); 
   
            
            
            //lvquxiaolunkuo
            if(contourArea(contours[b])>a)
            {
               if(P.x!=0 && P.y!=0)
                {                
                  contours1.push_back(contours[b]);
                  hierarchy1.push_back(hierarchy[b]);
                }
                else{
                  continue;
                }
            }
            else{
              continue;
            }
              
        }
    //guolvdiaokongdong
    vector<vector<Point>> contours3;  
    vector<Vec4i> hierarchy3;
    for(int b=0;b<contours1.size();b++) { 
            Point P=Point(contours1[b][0].x,contours1[b][0].y); 
            int inORout=0; 
            for(int j=0;j<contours1.size();j++)
                  {
                    if(j==b){continue;}
                    if(pointPolygonTest(contours1[j],P,false)==1)
                    {
                      inORout=1;
                      break;
                    }
                    else{continue;}
                  }
                  if(inORout==0)
                  {
                  contours3.push_back(contours1[b]);
                  hierarchy3.push_back(hierarchy1[b]);
                  }          
    }
     vector<vector<Point>> contours2;  
     vector<Vec4i> hierarchy2;
     if(c_size==0){
       contoursSky=contours3;  
       hierarchySky=hierarchy3;

       contours2=contours3;  
       hierarchy2=hierarchy3;        
       }
    else if(c_size!=0)
    {
    for(int b=0;b<contours3.size();b++) { 
            Point P=Point(contours3[b][1].x,contours3[b][1].y); 
            int inORout=0; 
            for(int j=0;j<contoursSky.size();j++)
                  { 
                    if(pointPolygonTest(contoursSky[j],P,false)==1)
                    {
                      inORout=1;
                      break;
                    }
                    else{continue;}
                  }
            if(inORout==0)
            {
             contours2.push_back(contours3[b]);
             hierarchy2.push_back(hierarchy3[b]);
            }          
      }  
    }
    if(contours2.size()==0){continue;}
    /*
    if(c_size==0){
            
            skyn+=1;
            }
            else if(c_size==2){
            
            roadn+=1;
            }
            else if(c_size==3){
            sidewalkn+=1;
            }
            else if(c_size==4){
            buildn+=1;
            }
            else if(c_size==5){
            fencen+=1;
            }
            else if(c_size==6){
            polen+=1;
            }
            else if(c_size==7){
            trafficsignn+=1;
            }
            else if(c_size==8){
            vegetationn+=1;
            }
            else if(c_size==9){
            trafficlightn+=1;
            }
            else if(c_size==10){
            roadlinen+=1;
            }
            */
    tuceng tc1;
    for(int i=0;i<contours2.size();i++)  
    {   
       /*
        if(c_size==0){
            skymianji += contourArea(contours2[i])/1724;
            
            }
            else if(c_size==2){
            skyroad += contourArea(contours2[i])/1724;
            
            }
            else if(c_size==3){
            skysidewalk += contourArea(contours2[i])/1724;
            }
            else if(c_size==4){
            skybuild += contourArea(contours2[i])/1724;
            }
            else if(c_size==5){
            skyfence += contourArea(contours2[i])/1724;
            }
            else if(c_size==6){
            skypole += contourArea(contours2[i])/1724;
            }
            else if(c_size==7){
            skytrafficsign += contourArea(contours2[i])/1724;
            }
            else if(c_size==8){
            skyvegetation += contourArea(contours2[i])/1724;
            }
            else if(c_size==9){
            skytrafficlight += contourArea(contours2[i])/1724;
            }
            else if(c_size==10){
            skyroadline += contourArea(contours2[i])/1724;
            }
            */
        int centerx=0,centery=0;
        int ptnum=contours2[i].size();
        node nd;
        //cout<<"lunkuodianshu"<<contours1[i].size()<<endl;
        //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数  
        for(int j=0;j<contours2[i].size();j++)   
        {  
            //绘制出contours向量内所有的像素点  
            Point P=Point(contours2[i][j].x,contours2[i][j].y);  
            Contours.at<uchar>(P)=255;  
            centerx+=P.x; 
            centery+=P.y;
        }  
        Point ct;
        ct.x=centerx/ptnum;
        ct.y=centery/ptnum;

        //tianjia zaosheng
       // ct.x+=rand()%301-150;
        //ct.y+=rand()%301-150;
        //cout<<"ct.x"<<rand()%401-200<<endl;
        nd.p=ct;
        nd.contours=contours2[i];
        nd.hierarchy=hierarchy2[i];
        nd.cls=c_size;
        nd.size=contourArea(contours2[i]);
        imagenodes.push_back(nd);
        tc1.size+=contourArea(contours2[i]);
        tc1.Centers.push_back(ct);
        //cout<<"lunkuozhongxin"<<ct.x<<" "<<ct.y<<endl;
        circle(imageContours,ct,2,Scalar(255,255,255));
        //绘制轮廓  
        drawContours(imageContours,contours2,i,Scalar(255),1,8,hierarchy2);
    } 
    if(contours2.size()!=0)
    {
      tc1.leibie=c_size;
      tc1.contourNum=contours2.size();   
      imagetucengs.push_back(tc1);   
    }

    }
    //cout<<"***********"<<imagenodes.size()<<endl;
    for(int nd=0;nd<imagenodes.size();nd++){
    //int nd=7;
    vector<Point> contour=imagenodes[nd].contours;
    Point c1=imagenodes[nd].p;
    int maxid=9999,maxid2=9999,maxid3=9999;
    
    float maxscore=999999,maxscore2=999999,maxscore3=999999;
    //cout<<"area size"<<sqrt(contourArea(contour))<<endl;
        for(int a=0;a<imagenodes.size();a++)
        {
        if(a==nd){continue;}
        float size=sqrt(contourArea(imagenodes[a].contours));
        Point c2=imagenodes[a].p;
        float dist1;
        dist1=(c1.x-c2.x)*(c1.x-c2.x)+(c1.y-c2.y)*(c1.y-c2.y);
        float dist=sqrt(dist1);
        float dist2=sqrt(dist);
        float score = dist*dist*sqrt(dist2)/size;
        if(score<maxscore)
        {
          maxscore3=maxscore2;
          maxid3=maxid2;
          maxscore2=maxscore;
          maxid2=maxid;
          maxscore=score;
          maxid=a;
        }
        else if(score<maxscore2)
        {
        maxscore3=maxscore2;
        maxid3=maxid2;
        maxscore2=score;
        maxid2=a;
        }
        else if(score<maxscore3)
        {
        maxscore3=score;
        maxid3=a;
        }
        else{continue;}    
        //cout<<"*dist*"<<dist<<endl;
        }
        line(imageContours,imagenodes[nd].p,imagenodes[maxid].p,Scalar(255),3);
        line(imageContours,imagenodes[nd].p,imagenodes[maxid2].p,Scalar(255),2);
        line(imageContours,imagenodes[nd].p,imagenodes[maxid3].p,Scalar(255),1);
        imagenodes[nd].twoDnbID.push_back(imagenodes[maxid].cls);
        imagenodes[nd].twoDnbID.push_back(imagenodes[maxid2].cls);
        imagenodes[nd].twoDnbID.push_back(imagenodes[maxid3].cls);
        imagenodes[nd].twoDnbSize.push_back(imagenodes[maxid].size);
        imagenodes[nd].twoDnbSize.push_back(imagenodes[maxid2].size);
        imagenodes[nd].twoDnbSize.push_back(imagenodes[maxid3].size);
        imagenodes[nd].twoDnbCenters.push_back(imagenodes[maxid].p);
        imagenodes[nd].twoDnbCenters.push_back(imagenodes[maxid2].p);
        imagenodes[nd].twoDnbCenters.push_back(imagenodes[maxid3].p);
  }
    allimagenodes.push_back(imagenodes);
    allimagetucengs.push_back(imagetucengs);
   // imshow("Contours Image",imageContours);
    ////imshow("erzhihua",gray);
   // waitKey(10000);

}

int main(int argc, char** argv) {
  for(int i=0;i<10;i++){
   l_5_id.push_back(0);
   l_5_flag.push_back(0); 
  }
  //chungjianfeilei rgb tongdao
  vector<int> sky; sky.push_back(128);sky.push_back(128);sky.push_back(128);classes.push_back(sky);//0 1
  vector<int> unlabeled; unlabeled.push_back(10);unlabeled.push_back(0);unlabeled.push_back(0);classes.push_back(unlabeled);//1 2
  vector<int> road; road.push_back(128);road.push_back(64);road.push_back(128);classes.push_back(road);//3 3
  vector<int> sidewalk; sidewalk.push_back(0);sidewalk.push_back(0);sidewalk.push_back(192);classes.push_back(sidewalk);//4 4
  vector<int> building; building.push_back(128);building.push_back(0);building.push_back(0);classes.push_back(building);//7 5
  vector<int> fence; fence.push_back(64);fence.push_back(64);fence.push_back(128);classes.push_back(fence);//9 6
  vector<int> pole; pole.push_back(192);pole.push_back(192);pole.push_back(128);classes.push_back(pole);//12 7
  vector<int> trafficsign; trafficsign.push_back(192);trafficsign.push_back(128);trafficsign.push_back(128);classes.push_back(trafficsign);//13 8
  vector<int> vegetation; vegetation.push_back(128);vegetation.push_back(128);vegetation.push_back(0);classes.push_back(vegetation);//14 9
  vector<int> trafficlight; trafficlight.push_back(0);trafficlight.push_back(128);trafficlight.push_back(128);classes.push_back(trafficlight);//18 10
  vector<int> roadline; roadline.push_back(0);roadline.push_back(175);roadline.push_back(0);classes.push_back(roadline);//21 11
  
 
  
  // Loading image filenames
  std::vector<std::string> filenames;
  getFilenames(argv[1], &filenames);
  unsigned nimages = filenames.size();

  
  // Processing the sequence of images
  for (unsigned i = 0; i < nimages; i++) {
  cv::Mat img1 = cv::imread(filenames[i]);
  //gettimeofday(&tv1, NULL);
  calculateNode(img1);
  //gettimeofday(&tv2, NULL);
	//计算用时
	//T = (tv2.tv_sec - tv1.tv_sec) * 1000 + (tv2.tv_usec - tv1.tv_usec) / 1000;
  //pjtime += T;
  //file<<i<<",";
  //file<<T<<"\n";
  //cout <<"Time spent on  : " << T << "ms" <<endl;
  }
  /* 
  cout <<"Time spent on  : " << pjtime/nimages << "ms" <<endl;
          
  cout <<"skymianji  : " << skymianji*1724/skyn<<endl;
  cout <<"skyroad  : " << skyroad*1724/roadn<<endl;
  cout <<"skysidewalk  : " << skysidewalk*1724/sidewalkn<<endl;
  cout <<"skybuild  : " << skybuild*1724/buildn<<endl;
  cout <<"skyfence  : " << skyfence*1724/fencen<<endl;
  cout <<"skypole  : " << skypole*1724/skyn<<endl;
  cout <<"skytrafficsign  : " << skytrafficsign*1724/skyn<<endl;
  cout <<"skyvegetation  : " << skyvegetation*1724/skyn<<endl;
  cout <<"skytrafficlight  : " << skytrafficlight*1724/skyn<<endl;
  cout <<"skyroadline  : " << skyroadline*1724/skyn<<endl;

 
 cout <<"skyn  : " << skyn<<endl;
  cout <<"roadn  : " << roadn<<endl;
  cout <<"sidewalkn  : " << sidewalkn<<endl;
  cout <<"buildn  : " << buildn<<endl;
  cout <<"fencen  : " << fencen<<endl;
  cout <<"polen  : " << polen<<endl;
  cout <<"trafficsignn  : " << trafficsignn<<endl;
  cout <<"vegetationn  : " << vegetationn<<endl;
  cout <<"trafficlightn  : " << trafficlightn<<endl;
  cout <<"roadlinen  : " << roadlinen<<endl;
  */
  //file.close();
 for(int ndNo=0;ndNo<813;ndNo++){
  //jisuanfenshu
  vector<node> temp=allimagenodes[ndNo];
  //cout<<"temp size"<<temp.size()<<endl;
  vector<tuceng> tucengtemp=allimagetucengs[ndNo];
  /*
  for(int a=0;a<tucengtemp.size();a++){
  cout<<"06"<<tucengtemp[a].leibie<<endl; 
  cout<<"06mianji"<<tucengtemp[a].size<<endl; 
  }
  */
  vector<float> daixuanfenshu;
  vector<int> daixuanid;
  double mkfyz=0.2;
  vector<double> mkfyzs;
  for(int im=0;im<allimagenodes.size();im++){
    if(im<813)continue;

    //gettimeofday(&tv1, NULL);

    float tucengflag=0;
    vector<node> temp1=allimagenodes[im];
    vector<tuceng> tucengtemp1=allimagetucengs[im];
    float imagescore=0;
    
    int min_temp_size=0;
    if(temp.size()<temp1.size()){
      min_temp_size=temp.size();
    }
    else{
      min_temp_size=temp1.size();
    }
   
  
    
   // if(tucengcha<0){tucengcha=0-tucengcha;}
    //if(tucengcha>0){tucengflag=1;}
    
  
    //cout<<"tucengflag"<<tucengflag<<endl;
    float tcxsx=1;
    if(tucengflag==9999){
      imagescore=0;
    }
    
    else{
    vector<Point> cps;
    vector<Point> cps1;
    //jisuanmeiyizhangtupiandefenshu
    for(int tp=0;tp<temp.size();tp++)
    {
      node tempnode=temp[tp];
      vector<int> tdnb=tempnode.twoDnbID;
      vector<float> szs=tempnode.twoDnbSize;
      vector<Point> Pts=tempnode.twoDnbCenters;
      float mj=tempnode.size;
      Point ctr=tempnode.p;
      //cout<<"****mobanlinju"<<tdnb[0]<<" "<<tdnb[1]<<" "<<tdnb[2]<<endl;
      float bestscore=0;
      Point cp;
      Point cp1;
      float leibieRatio=0;int leibie=tempnode.cls;
      switch (leibie)
      {
      case 0:leibieRatio=1;break;
      case 1:leibieRatio=1;break;
      case 2:leibieRatio=1;break;
      case 3:leibieRatio=1;break;
      case 4:leibieRatio=1;break;
      case 5:leibieRatio=1.2;break;
      case 6:leibieRatio=1.1;break;
      case 7:leibieRatio=1.3;break;
      case 8:leibieRatio=1;break;
      case 9:leibieRatio=1.3;break;
      case 10:leibieRatio=1.1;break;
      case 11:leibieRatio=1.1;break;

      default:
        break;
      }
      
      for(int tp1=0;tp1<temp1.size();tp1++){
      node tempnode1=temp1[tp1];  
      vector<int> tdnb1=tempnode1.twoDnbID;
      vector<float> szs1=tempnode1.twoDnbSize;
      vector<Point> Pts1=tempnode1.twoDnbCenters;
      float mj1=tempnode1.size;
      Point ctr1=tempnode1.p;
      

      //cout<<"jiancelinju"<<tdnb1[0]<<" "<<tdnb1[1]<<" "<<tdnb1[2]<<endl;
       if(tempnode.cls!=tempnode1.cls){continue;}
       else{
            //jisuanmianjibi
              float areaRatio1=mj/mj1;
              float areaRatio=0;
              if(areaRatio1>1){
                areaRatio=1/areaRatio1;
              }
              else{
              areaRatio=areaRatio1;
              }
              
              if(areaRatio>0.95)
              {
                areaRatio=1;
              }
              else if(areaRatio<0.70)
              {
                areaRatio=0;
                 //continue;
              }
              
              //cout<<areaRatio<<endl;
              float disRatio=0;
              float ll=(ctr.x-ctr1.x)*(ctr.x-ctr1.x)+(ctr.y-ctr1.y)*(ctr.y-ctr1.y);
              float l=sqrt(ll);
              if(l<=(0.32*h))
              {
               disRatio=1; 
              }
              else if(l>=(0.75*h))
              {
                 disRatio=0;
              }
              else{
               disRatio=1-(l-0.32*h)/(0.43*h);
              } 
              if(disRatio==0||areaRatio==0)continue;
              
             
            

            float score;
            float num=0,num1=0;
            float s1=0,s2=0;            
            for(int a=0;a<3;a++)
            {
              float edgeScore=0;
              int clas=tdnb[a];
              float sz=szs[a];
              Point ctpt=Pts[a];

              if(a==0){s1=1;}
              else if(a==1){s1=1;}
              else if(a==2){s1=1;}

              for(int b=0;b<3;b++){
               if(b==0){s2=1;}
               else if(b==1){s2=1;}
               else if(b==2){s2=1;}

               int clas1=tdnb1[b]; 
               float sz1=szs1[b];
               Point ctpt1=Pts1[b];
               float diss=(ctpt.x-ctpt1.x)*(ctpt.x-ctpt1.x)+(ctpt.y-ctpt1.y)*(ctpt.y-ctpt1.y);
               float dis=sqrt(diss);
               float Ratio1=sqrt(sz/sz1);
               float Ratio=0;
               if(Ratio1>1){Ratio=1/Ratio1;}
               else{Ratio=Ratio1;}
               
               //cout<<"ratio"<<Ratio<<endl;
               if(clas==clas1){                
                tdnb1[b]=999;
                edgeScore=s1*s2;
                //score+=0.333;
                num+=edgeScore;
                num1++;
                break;
               }
               else{continue;}
              }  

            }
            //cout<<"***num***"<<num<<endl;
            score = num*0.33333;
            score=score*sqrt(areaRatio)*disRatio;
            //if(num1==3){score=score*1;}
            //else if(num1==2){score=score*0.8;}
            //else if(num1==1){score=score*0.3;}
            //else if(num1==0){score=score*0;}
            if(score>bestscore){bestscore=score;cp=tempnode.p;cp1=tempnode1.p;}
            //bestscore=1;          
         
          
        }
        
      }

      if(bestscore>0){
        cps.push_back(cp);
        cps1.push_back(cp1);
        imagescore+=bestscore;
      }
      
    }
    int per=0;

    if(temp1.size()>=temp.size()){
      per=temp1.size();
    }
    else{
     per=temp.size();
    }
     //*******jisuanchulexiangsixing*********
      imagescore=imagescore/per;
     //imagescore=imagescore/per*tcxsx;
    //*************************************************************
  }    
       
      
        if(imagescore>0.00){
        //cout<<"yuanimageid"<<0<<"imageid"<<im<<"score"<<imagescore<<endl;
        daixuanfenshu.push_back(imagescore);
        daixuanid.push_back(im);
        }  
    //cout<<"yuanimageid"<<0<<"imageid"<<im<<"score"<<imagescore<<endl;  

    //gettimeofday(&tv2, NULL);
	//计算用时
	//T = (tv2.tv_sec - tv1.tv_sec) * 1000000 + (tv2.tv_usec - tv1.tv_usec) ;
  //pjtime += T;
  //file<<im<<",";
  //double a=T;
  //file<<a<<"\n";
  //cout <<"Time spent on  : " << a << "ms" <<endl;
  }
  //cout <<"Time spent on  : " << pjtime/911<< "ms" <<endl;
  //pjtime1+= pjtime/911;
  //pjtime=0;
  //file.close();
  
  
       
          //cout<<"max"<<mkfyz<<endl;
      
     if(daixuanid.size()==0){
          cout<<"yuanimageid"<<ndNo<<"meiyouhuihuan"<<endl;
        }
        else{
          float maxScore=daixuanfenshu[0];
          int maxId=daixuanid[0];
          int maxId1=daixuanid[0];
          int maxId2=daixuanid[0];
          int maxId3=daixuanid[0];
          int maxId4=daixuanid[0];
          int maxId5=daixuanid[0];
          int maxId6=daixuanid[0];
          for(int x=0;x<daixuanfenshu.size();x++){
             if(daixuanfenshu[x]>maxScore){
               maxScore=daixuanfenshu[x];
               maxId6=maxId5; maxId5=maxId4;maxId4=maxId3;maxId3=maxId2;maxId2=maxId1;maxId1=maxId;
               maxId=daixuanid[x];
               
               
               
               
              
               

             }
             else{
               continue;
             }    
          }
        //cout<<"yuan_image"<<ndNo<<"imageid"<<maxId-813<<"score"<<maxScore<<endl;
        cout<<"query_imageid:"<<ndNo<<" ref_imageid:"<<maxId-813<<endl;
        /*
        file<<maxId-813<<",";
        file<<maxId1-813<<",";
        file<<maxId2-813<<",";
        file<<maxId3-813<<",";
        file<<maxId4-813<<",";
        file<<maxId5-813<<",";
        file<<maxId6-813<<"\n";
        */
         for(int i=0;i<911;i++){
                if(i==maxId-813){
                  if(i!=910){
                    file<<maxScore*100<<",";
                  }
                  else{
                    file<<maxScore*100<<"\n";
                  }
                  
                }
                else{
                  if(i!=910){
                    file<<0<<",";
                  }
                  else{
                    file<<0<<"\n";
                  }
                  
                }
                
              }
              
            
        
        }  
       
        
}
  file.close();
  
  return 0;
}

