#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "linear.h"


void predictLinear(double *Y, double *X, double *W, int *decisionValues, double *DecisionValues, int *proba, double *Probabilities, int *nbClass, int *nbDim, int *nbSamples, int *sparse, int *rowindex, int *colindex, double *bias, int *labels, int *type);


/**
 * Function: predictLinear
 *
 * Author: Thibault Helleputte
 *
 */
void predictLinear(double *Y, double *X, double *W, int *decisionValues, double *DecisionValues, int *proba, double *Probabilities, 
        int *nbClass, int *nbDim, int *nbSamples, int *sparse, int *rowindex, int *colindex, double *bias, int *labels, int *type){
	
	struct feature_node *x;
	struct parameter par;
	struct model model_;
	int i, j, n, predict_label;
	double *prob_estimates=NULL;
	double *decision_values=NULL;
	
	// RECONSTRUCT THE (REQUIRED) PARAMETERS
	par.solver_type=*type;
	
	// RECONSTRUCT THE MODEL
	model_.nr_class=*nbClass;
	model_.nr_feature=*nbDim;
	model_.bias=*bias;
	model_.param=par;
	if(*bias>=0)
		n=*nbDim+1;
	else
		n=*nbDim;
	
	model_.w=W;

	model_.label=labels;
	
	x = (struct feature_node *) Calloc(n+1,struct feature_node);
	
	if(*proba)
	{
		if(!check_probability_model(&model_))
		{
			Rprintf("Error: probability output is only supported for logistic regression.\n");
			return;
		}
		prob_estimates = (double *) Calloc(*nbClass,double);
	}
	
	if(*decisionValues)
		decision_values = (double *) Calloc(*nbClass,double);

    int totalK = 0;
	// PREDICTION PROCESS	
	for(i=0; i<*nbSamples; i++){
        if(*sparse > 0){
            // trying to handle the sparse matrix
            int nnz = rowindex[i+1]-rowindex[i];
            for(j = 0; j<nnz; j++){
                x[j].value = X[totalK];
                x[j].index = colindex[totalK];
                totalK++;
            }
        } else {
            int k = 0;
            for(j=0; j<*nbDim; j++){
                if(X[(*nbDim*i)+j]!=0){
                    x[k].value = X[(*nbDim*i)+j];
                    x[k].index = j+1;
                    k++;
                }
            }
            j = k;
        }

		if(model_.bias>=0){
			x[j].index = n;
			x[j].value = model_.bias;
			j++;
		}
		x[j].index = -1;

		if(*proba){
			predict_label = predict_probability(&model_,x,prob_estimates);
			for(j=0;j<model_.nr_class;j++)
				Probabilities[model_.nr_class*i+j]=prob_estimates[j];
		}
		else{
			predict_label = predict(&model_,x);
		}
		Y[i]=predict_label;

		if(*decisionValues){
			predict_label = predict_values(&model_,x,decision_values);
			for(j=0;j<model_.nr_class;j++)
				DecisionValues[model_.nr_class*i+j]=decision_values[j];
		}
	}
	if(*proba)
		Free(prob_estimates);
	if(*decisionValues)
		Free(decision_values);
	return;
}

