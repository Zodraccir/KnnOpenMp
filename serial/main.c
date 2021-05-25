#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUMBER_ATTRS 4
#define NUMBER_OF_CLUSTER 3
#define NUMBER_OF_SPLIT 4

struct row {
	double attrs[NUMBER_ATTRS];
	int classification;
};

double EuclideanDistance(struct row x,struct row x0){
	long double d=0;
	long double df;
	for(int i=0;i<NUMBER_ATTRS;i++){
		df=x.attrs[i]-x0.attrs[i];
		d=d+ df*df; 
	}
    return sqrt(d);
}



struct row * loadDataset(char* filename,int *size,int isTraining){

 	struct row *set;
 	freopen(filename,"r",stdin);
 	
 	int dim=4;
 	char s[100];
	
	if (!(set = malloc(dim * sizeof(struct row)))) {
		perror("Errore allocazione memoria\n");
		//return 1;
	}

	*size = 0;
	while (fgets(s, sizeof(s), stdin)) {
		if(isTraining==0){
		sscanf(s, "%le %le %le %le", &set[*size].attrs[0], &set[*size].attrs[1], &set[*size].attrs[2], &set[*size].attrs[3]);
		
		}
		else if(isTraining==1){
		sscanf(s, "%le %le %le %le %d", &set[*size].attrs[0], &set[*size].attrs[1], &set[*size].attrs[2], &set[*size].attrs[3], &set[*size].classification);
		}
	
		*size=*size+1;
		/* 
		 * se si supera la dimensione del vettore 
		 * si raddoppia la dimensione e si rialloca un nuovo vettore
		 * di tale dimensione
		 */
		if (*size >= dim) {
			dim *= 2;
			if (!(set = realloc(set, dim * sizeof(struct row)))) {
				perror("Errore allocazione memoria\n");
				//exit();
				//return 1;
			}
		}
	}

	return set;

}


struct row * splitDataSet(struct row * trainingSet,int start, int end){
	int size=end-start;
	//printf("memoria allocata %d\n",size);
	struct row *returSet;
	if (!(returSet = malloc(size * sizeof(struct row)))) {
		perror("Errore allocazione memoria\n");
		//return 1;
	}
	int n=0;
	for(int i=start;i<end;i++){
	
		for(int j=0;j<NUMBER_ATTRS;j++){
			returSet[n].attrs[j]=trainingSet[i].attrs[j];
		}
		
		
		
		returSet[n].classification=trainingSet[i].classification;
		n++;
	}
	return returSet;

}

struct row * splitDataSetTraining(struct row * trainingSet,int start, int end,int *sizeNewTraining,int sizeTraining){
	*sizeNewTraining=sizeTraining-(end-start);
	//printf("memoria allocata %d\n",*sizeNewTraining);
	struct row *returSet;
	if (!(returSet = malloc(*sizeNewTraining * sizeof(struct row)))) {
		perror("Errore allocazione memoria\n");
		//return 1;
	}
	int n=0;
	for(int i=0;i<sizeTraining;i++){
		if(i<start || i>=end){
		for(int j=0;j<NUMBER_ATTRS;j++){
			returSet[n].attrs[j]=trainingSet[i].attrs[j];
		}
		returSet[n].classification=trainingSet[i].classification;
		n++;
		}
		
	}
	return returSet;
}


void classifyPointFromDistance(struct row * trainingSet, struct row * testSet, double* distance,int k ,int sizeTraining){
	
   struct row knn[k]; 
        
        int point[k];
        
        for(int i = 0 ; i<k ; i++){
            double min=99999999;
            for(int j = 0 ; j<sizeTraining ; j++){
                if(distance[j]<min){
                    min=distance[j];
                    point[i]=j;
                }
            }
            distance[point[i]]=99999999;
            knn[i]=trainingSet[point[i]];            
        }
  
	//count number of occurrences of of classification, work with index
	int NumberClassification[NUMBER_OF_CLUSTER]={0};
        for(int e=0;e<k;e++){
            NumberClassification[knn[e].classification-1]+=1;
        }

        int max=0;
        int classification;

	//pick the max occurrency of classification with index
        for(int h=0;h<NUMBER_OF_CLUSTER;h++){
            if(NumberClassification[h]>max){
                max=NumberClassification[h];
                classification=h;
            }
        }
	int class=classification+1;
//      printf("index %d CLASSIFICATED--> %d, Numero thread: %d \n",ii, class, omp_get_thread_num());        
	
		
		testSet->classification=class;
   
}


void performKnn(struct row * trainingSet, struct row * testSet,int k,int sizeTraining,int sizeTest){


     //float distance[sizeTest][sizeTraining];


	//int r = 3, c = 4, i, j, count;
  
    double **distance = (double **)malloc(sizeTest * sizeof(double *));
    for (int i=0; i<sizeTest; i++)
         distance[i] = (double *)malloc(sizeTraining * sizeof(double));
     

     for(int ii=0;ii<sizeTest;ii++){
     
	for(int t = 0 ; t<sizeTraining ; t++){
		distance[ii][t]=EuclideanDistance(testSet[ii],trainingSet[t]);
        }
      }
      
      

      for(int ii=0;ii<sizeTest;ii++){          
	classifyPointFromDistance(trainingSet, &testSet[ii], distance[ii],k,sizeTraining);
    }
   
   for (int r = 0; r < sizeTest; r++) {
        free(distance[r]);
    }
 
   free(distance);
}


void printPixels(struct row * pixels,int sizeTraining){

	for(int i=0;i<sizeTraining;i++) {

		for(int j=0;j<NUMBER_ATTRS;j++){
			printf("%.2lf ",pixels[i].attrs[j]);
		}
		printf("%d\n",pixels[i].classification);

	}
}



double validateKnn(struct row * trainingSet, struct row * crossValidationSet,int sizeCross,int start,int end){
	int counter=0;
	for(int i=0;i<sizeCross;i++){
		if(crossValidationSet[i].classification==trainingSet[start+i].classification){
			counter++;
			}
	}
	
	return (double) counter/ (double) sizeCross;
}



int main(int argc, char *argv[]){



 

	int sizeTraining=0;
	int sizeTest =0;
    	
	
	

	struct row *trainingSet = loadDataset("irisPD.txt", &sizeTraining,1);
	//printPixels(trainingSet,sizeTraining);
	
	struct row *testSet = loadDataset("test.txt", &sizeTest,0);
	
	
	int K=50;
	
	double bestK[K-1];
	

	for(int k=1;k<K;k++){
	
	double meanK[NUMBER_OF_SPLIT];
			
	for(int split=0;split<NUMBER_OF_SPLIT;split++){
	
	
		
		int sizeCrossValidationSet =0;
		
		int startSplitCorr=(int) sizeTraining/NUMBER_OF_SPLIT*split;
		int endSplitCorr=((int) sizeTraining/NUMBER_OF_SPLIT)*(split+1);
		
		int dimCorr=endSplitCorr-startSplitCorr;
		
		
		int sizeNewTraining;
		struct row *newTraningSet = splitDataSetTraining(trainingSet,startSplitCorr,endSplitCorr,&sizeNewTraining,sizeTraining);
		struct row *crossValidationSet=splitDataSet(trainingSet,startSplitCorr,endSplitCorr);
		
		
		

		//printf("--------------dopo--------------");

		performKnn(newTraningSet,crossValidationSet,k,sizeNewTraining,dimCorr);
	
		double value=validateKnn(trainingSet,crossValidationSet,dimCorr,startSplitCorr,endSplitCorr);
		
		
		//printf("k:%d value 1/4 %lf\n",k,value);
		
		meanK[split]=value;
		
		free(newTraningSet);
		free(crossValidationSet);
	}

	double sumMeanK=0;
	for(int l=0;l<NUMBER_OF_SPLIT;l++)
		sumMeanK=sumMeanK+meanK[l];
		bestK[k-1]=sumMeanK/4;
	
	}
	
	
	double max=bestK[0];
	int indexMax=0;
	for(int j=0;j<K-1;j++){
		printf("%lf\n",bestK[j]);
		if(bestK[j]>max){
			max=bestK[j];
			indexMax=j;
		}
	}
	
	printf("best %lf %d \n",max,indexMax);
		
		
	performKnn(trainingSet,testSet,indexMax+1,sizeTraining,sizeTest);
	
	
	printPixels(testSet,sizeTest);
	
	free(trainingSet);
	
	free(testSet);
	
	
}
