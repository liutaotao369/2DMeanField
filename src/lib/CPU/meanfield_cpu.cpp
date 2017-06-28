/*
 Copyright (c) 2016, Jack Miles Hunt
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
 * Neither the name of Jack Miles Hunt nor the
	  names of contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Jack Miles Hunt BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "meanfield_cpu.h"

using namespace MeanField::CPU;
using namespace MeanField::Filtering;

CRF::CRF(int width, int height, int dimensions, float spatialSD,
	float bilateralSpatialSD, float bilateralIntensitySD, bool separable) :
	width(width), height(height), dimensions(dimensions),
	spatialWeight(1.0), bilateralWeight(1.0),
	spatialSD(spatialSD), bilateralSpatialSD(bilateralSpatialSD),
	bilateralIntensitySD(bilateralIntensitySD),
	separable(separable),
	QDistribution(new float[width * height * dimensions]()),
	QDistributionTmp(new float[width * height * dimensions]()),
	pottsModel(new float[dimensions * dimensions]()),
	gaussianOut(new float[width * height * dimensions]()),
	bilateralOut(new float[width * height * dimensions]()),
	aggregatedFilters(new float[width * height * dimensions]()),
	filterOutTmp(new float[width * height * dimensions]()),
#ifndef WITH_PERMUTOHEDRAL
	spatialKernel(new float[KERNEL_SIZE]()),
	bilateralSpatialKernel(new float[KERNEL_SIZE]()),
	bilateralIntensityKernel(new float[KERNEL_SIZE]()) {
#else
	spatialKernel(new float[2 * width * height]()),
	bilateralKernel(new float[5 * width * height]()),
	spatialNorm(new float[width * height]()),
	bilateralNorm(new float[width * height]()),
	onesImage(new float[width * height]()),
	spatialLattice(new Permutohedral::ModifiedPermutohedral()),
	bilateralLattice(new Permutohedral::ModifiedPermutohedral()) {
#endif

	//Initialise potts model.
	for (int i = 0; i < dimensions; i++) {
		for (int j = 0; j < dimensions; j++) {
			pottsModel[i * dimensions + j] = (i == j) ? -1.0 : 0.0;
		}
	}

	//Initialise kernels.
#ifndef WITH_PERMUTOHEDRAL
	generateGaussianKernel(spatialKernel.get(), KERNEL_SIZE, spatialSD);
	generateGaussianKernel(bilateralSpatialKernel.get(), KERNEL_SIZE, bilateralSpatialSD);
	generateGaussianKernel(bilateralIntensityKernel.get(), KERNEL_SIZE, bilateralIntensitySD);
#else
	//Generate ones image for computing normalisers.
	float *onesData = onesImage.get();
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < width * height; i++) {
		onesData[i] = 1.0;
	}

	//Generate spatial kernel.
	float *spatialKernelData = spatialKernel.get();
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < width * height; i++) {
		generateGaussianKernelPoint(spatialKernelData, i, width, spatialSD);
	}

	//Initialise spatial lattice and compute norm image.
	float *spatialNormData = spatialNorm.get();
	spatialLattice->init(spatialKernelData, 2, width, height);
	spatialLattice->compute(spatialNormData, onesData, 1);

	//Add small constant to norm.
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < width * height; i++) {
		spatialNormData[i] += 1e-20f;
	}
#endif	
}

CRF::~CRF() {
	//
}

void CRF::runInference(const unsigned char *image, const float *unaries, int iterations) {
#ifdef WITH_PERMUTOHEDRAL
	//Generate bilateral kernel.
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < width * height; i++) {
		generateBilateralKernelPoint(bilateralKernel.get(), image, i, width, bilateralSpatialSD, bilateralIntensitySD);
	}

	//Reset bilateral lattice.
	bilateralLattice.reset(new Permutohedral::ModifiedPermutohedral());

	//Initialise bilateral lattice and compute norm image.
	bilateralLattice->init(bilateralKernel.get(), 5, width, height);
	bilateralLattice->compute(bilateralNorm.get(), onesImage.get(), 1);

	//Add small constant to norm.
	float *bilateralNormData = bilateralNorm.get();
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < width * height; i++) {
		bilateralNormData[i] += 1e-20f;
	}
#endif//End setup for bilateral lattice.

	//Run inference.
	runInferenceIteration(image, unaries);
	for (int i = 0; i < iterations - 1; i++) {
		runInferenceIteration(image, QDistribution.get());
	}
}

void CRF::runInferenceIteration(const unsigned char *image, const float *unaries) {
	filterGaussian(unaries);
	filterBilateral(unaries, image);
	weightAndAggregate();
	applyCompatabilityTransform();
	subtractQDistribution(unaries, aggregatedFilters.get(), QDistributionTmp.get());
	applySoftmax(QDistributionTmp.get(), QDistribution.get());
}

void CRF::setSpatialWeight(float weight) {
	spatialWeight = weight;
}

void CRF::setBilateralWeight(float weight) {
	bilateralWeight = weight;
}

void CRF::reset() {
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < width * height * dimensions; i++) {
		QDistribution[i] = 0.0;
		QDistributionTmp[i] = 0.0;
		gaussianOut[i] = 0.0;
		bilateralOut[i] = 0.0;
	}
}


const float *CRF::getQ() {
	return QDistribution.get();
}

void CRF::filterGaussian(const float *unaries) {
#ifndef WITH_PERMUTOHEDRAL
	if (!separable) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				applyGaussianKernel(spatialKernel.get(), unaries, gaussianOut.get(), spatialSD, dimensions, j, i, width, height);
			}
		}
	}
	else {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				applyGaussianKernelX(unaries, filterOutTmp.get(), spatialKernel.get(), spatialSD,
					dimensions, j, i, width, height);
			}
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				applyGaussianKernelY(filterOutTmp.get(), gaussianOut.get(), spatialKernel.get(), spatialSD,
					dimensions, j, i, width, height);
			}
		}
	}
#else
	spatialLattice->compute(gaussianOut.get(), unaries, dimensions);
	//Normalise the output.
	float *gaussianData = gaussianOut.get();
	const float *normData = spatialNorm.get();
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < width * height; i++) {
		for (int j = 0; j < dimensions; j++) {
			int idx = i * dimensions + j;
			gaussianData[idx] /= normData[i];
		}
	}
#endif
}

void CRF::filterBilateral(const float *unaries, const unsigned char *image) {
#ifndef WITH_PERMUTOHEDRAL
	if (!separable) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				applyBilateralKernel(bilateralSpatialKernel.get(), bilateralIntensityKernel.get(), unaries,
					image, bilateralOut.get(), bilateralSpatialSD, bilateralIntensitySD,
					dimensions, j, i, width, height);
			}
		}
	}
	else {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				applyBilateralKernelX(unaries, filterOutTmp.get(), image, bilateralSpatialKernel.get(),
					bilateralIntensityKernel.get(), bilateralSpatialSD, bilateralIntensitySD,
					dimensions, j, i, width, height);
			}
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				applyBilateralKernelY(filterOutTmp.get(), bilateralOut.get(), image, bilateralSpatialKernel.get(),
					bilateralIntensityKernel.get(), bilateralSpatialSD, bilateralIntensitySD,
					dimensions, j, i, width, height);
			}
		}
	}
#else
	//bilateralLattice->compute(bilateralOut.get(), unaries, dimensions);
	//Normalise the output.
	float *bilateralData = bilateralOut.get();
	const float *normData = bilateralNorm.get();
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < width * height; i++) {
		for (int j = 0; j < dimensions; j++) {
			int idx = i * dimensions + j;
			//bilateralData[idx] /= normData[i];
		}
	}
#endif
}

void CRF::weightAndAggregate() {
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < height * width; i++) {
		for (int k = 0; k < dimensions; k++) {
			int idx = i * dimensions + k;
			weightAndAggregateIndividual(gaussianOut.get(), bilateralOut.get(), aggregatedFilters.get(),
					spatialWeight, bilateralWeight, idx);
		}
	}
}

void CRF::applyCompatabilityTransform() {
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < height * width; i++) {
		applyCompatabilityTransformIndividual(pottsModel.get(), aggregatedFilters.get(), i, dimensions);
	}
}

void CRF::subtractQDistribution(const float *unaries, const float *QDist, float *out) {
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < width * height * dimensions; i++) {
		out[i] = unaries[i] - QDist[i];
	}
}

void CRF::applySoftmax(const float *QDist, float *out) {
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int i = 0; i < height * width; i++) {
		applySoftmaxIndividual(QDist, out, i, dimensions);
	}
}
