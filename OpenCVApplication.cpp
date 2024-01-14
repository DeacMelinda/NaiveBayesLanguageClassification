// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <string.h>
#include <unordered_map>

#define THRESHOLD 1

void toLower(std::string& text) {
	for (char& c : text) {
		c = tolower(c);
	}
}

int readTextFile(FILE* file, std::vector<std::string>& data, std::vector<std::string>& labels, int* numSamples) {

	size_t lineLength = 512;
	char* line = (char*)malloc(lineLength * sizeof(char));

	std::vector<std::string> tempData;
	std::vector<std::string> tempLabels;

	int count = 0;

	while (fgets(line, lineLength, file) != NULL) {
		char* label = strtok(line, "|");
		char* text = line + strlen(label) + 1;

		text[strcspn(text, "\n")] = '\0';

		tempData.push_back(text);
		tempLabels.push_back(label);

		toLower(tempData[count]);

		count++;
	}

	data = tempData;
	labels = tempLabels;
	*numSamples = count;

	free(line);

	return 0;
}

std::vector<std::string> initializeLetterPairs()
{
	std::vector<std::string> letterPairs;
	for (char c1 = 'a'; c1 <= 'z'; c1++)
	{
		for (char c2 = 'a'; c2 <= 'z'; c2++)
		{
			std::string pair = std::string(1, c1) + std::string(1, c2);
			letterPairs.push_back(pair);
		}

		letterPairs.push_back(std::string(1, c1) + " ");
		letterPairs.push_back(" " + std::string(1, c1));
	}

	letterPairs.push_back("  ");

	return letterPairs;
}

std::pair<Mat_<double>, Mat_<double>> train(const std::vector<std::string> data, std::vector<std::string> labels, int numSamples, std::vector<std::string> letterPairs, int C)
{
	const int d = letterPairs.size(); // number of features
	Mat_<uchar> X(numSamples, d); 
	Mat_<uchar> y(numSamples, 1);
	Mat_<double> priors(C, 1);
	Mat_<double> likelihood(C, d);

	X.setTo(0);

	int nrOfEnglishSamples = 0;
	int nrOfDutchSamples = 0;

	int X_row = 0;

	for (int i = 0; i < data.size(); i++)
	{
		std::string str = data[i];
		for (size_t i = 0; i < str.length() - 1; i++)
		{
			std::string pair = str.substr(i, 2);

			auto it = std::find(letterPairs.begin(), letterPairs.end(), pair);

			if (it != letterPairs.end())
			{
				int index = std::distance(letterPairs.begin(), it);
				X(X_row, index)++;
			}
		}
		if (labels[i] == "en")
		{
			nrOfEnglishSamples++;
			y(X_row, 0) = 0;
		}
		else
		{
			nrOfDutchSamples++;
			y(X_row, 0) = 1;
		}
		X_row++;
	}

	priors(0, 0) = (1.0 * nrOfEnglishSamples) / numSamples;
	priors(1, 0) = (1.0 * nrOfDutchSamples) / numSamples;

	// likelihood for English
	for (int feat = 0; feat < d; feat++)
	{
		int nr = 0;
		for (int k = 0; k < X.rows; k++)
		{
			if (X(k, feat) > THRESHOLD && y(k, 0) == 0) // the feature is over a given threshold (appeard at least once) and the class of the row is english (0)
			{
				nr++;
			}
		}
		likelihood(0, feat) = (nr + 1.0) / static_cast<double>(nrOfEnglishSamples + C);
	}

	// likelihood for Dutch
	for (int feat = 0; feat < d; feat++)
	{
		int nr = 0;
		for (int k = 0; k < X.rows; k++)
		{
			if (X(k, feat) > THRESHOLD && y(k, 0) == 1) // the feature is over a given threshold (appeard at least once) and the class of the row is dutch (1)
			{
				nr++;
			}
		}
		likelihood(1, feat) = (nr + 1.0) / static_cast<double>(nrOfDutchSamples + C);
	}

	std::cout << "X = " << X.row(1) << std::endl;
	std::cout << "y = " << y.row(1) << std::endl;

	std::cout << "Engl: " << nrOfEnglishSamples << "\nDutch: " << nrOfDutchSamples << std::endl;

	return { priors, likelihood };
}

double computeAccuracy(const Mat_<int> confusionMatrix) {
	int numClasses = confusionMatrix.rows;
	int totalExamples = sum(confusionMatrix)[0];

	int correctPredictions = 0;
	for (int i = 0; i < numClasses; i++) {
		correctPredictions += confusionMatrix(i, i);
	}

	double accuracy = static_cast<double>(correctPredictions) / totalExamples;
	return accuracy;
}

int classifyBayes(std::string text, Mat_<double> priors, Mat_<double> likelihood, std::vector<std::string> letterPairs)
{
	Mat_<uchar> text_X(1, letterPairs.size());

	text_X.setTo(0);
	for (size_t i = 0; i < text.length() - 1; i++)
	{
		std::string pair = text.substr(i, 2);

		auto it = std::find(letterPairs.begin(), letterPairs.end(), pair);

		if (it != letterPairs.end())
		{
			int index = std::distance(letterPairs.begin(), it);
			text_X(0, index)++;
		}
	}
	
	Mat_<double> priorForClass(priors.rows, 1);
	for (int c = 0; c < priors.rows; c++)
	{
		priorForClass(c, 0) = 0;
		double sumOfLikelihoods = 0;
		for (int feat = 0; feat < text_X.cols; feat++)
		{
			if (text_X(0, feat) > THRESHOLD)
			{
				sumOfLikelihoods += log(likelihood(c, feat));
			}
			else
			{
				sumOfLikelihoods += log(1 - likelihood(c, feat));
			}
		}
		priorForClass(c, 0) = log(priors(c, 0)) + sumOfLikelihoods;
	}

	int maxClass = 0;
	double maxVal = -INFINITY;

	for (int i = 0; i < priors.rows; i++)
	{
		if (priorForClass(i, 0) > maxVal)
		{
			maxVal = priorForClass(i, 0);
			maxClass = i;
		}
	}

	return maxClass;
}

void test(const std::vector<std::string> data, std::vector<std::string> labels, int numSamples, std::vector<std::string> letterPairs, Mat_<double> priors, Mat_<double> likelihood, int C)
{
	Mat_<double> confusionMatrix = Mat::zeros(C, C, CV_32FC1);

	int nrSamples = 0;

	for (int i = 0; i < data.size(); i++)
	{
		std::string str = data[i];
		int predictedClass = classifyBayes(str, priors, likelihood, letterPairs);
		confusionMatrix(labels[i] == "en" ? 0 : 1, predictedClass)++;
		nrSamples++;
		std::cout << "For: " << i << " the class is " << labels[i] << " and it predicted " << (predictedClass == 0 ? "en" : "nl") << std::endl;
	}

	double accuracy = computeAccuracy(confusionMatrix);
	std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
	
	std::cout << confusionMatrix << std::endl;
	namedWindow("ConfMat", WINDOW_KEEPRATIO);
	confusionMatrix /= 1.0 * nrSamples / static_cast<double>(C);
	//std::cout << confusionMatrix << std::endl;
	imshow("ConfMat", confusionMatrix);
}

void project() {

	std::vector<std::string> letterPairs = initializeLetterPairs();

	std::cout << "letterPairs" << std::endl;

	for (int i = 0; i < letterPairs.size(); i++)
	{
		std::cout << letterPairs[i] << std::endl;
	}

	const int C = 2; // 2 languages
	const int d = letterPairs.size(); // number of features - total number of possible pairs

	Mat_<double> likelihood(C, d);
	Mat_<double> priors(C, 1);

	FILE* file = fopen("C:\\Users\\Melinda\\Desktop\\AnIV_SemI\\PRS\\Project\\OpenCVApplication-VS2015_31_basic\\eng_nl_texts_train.txt", "r");

	if (file == NULL) {
		fprintf(stderr, "Error opening file.\n");
		exit(0);
	}

	std::vector<std::string> data;
	std::vector<std::string> labels;

	int numSamples = 0;

	if (readTextFile(file, data, labels, &numSamples) != 0) {
		fprintf(stderr, "Error reading file.\n");
		exit(0);
	}

	auto trainingResult = train(data, labels, numSamples, letterPairs, C);

	priors = trainingResult.first;
	likelihood = trainingResult.second;

	std::cout << "Priors: " << priors << std::endl;

	FILE* testFile = fopen("C:\\Users\\Melinda\\Desktop\\AnIV_SemI\\PRS\\Project\\OpenCVApplication-VS2015_31_basic\\eng_nl_texts_test.txt", "r");

	if (testFile == NULL) {
		fprintf(stderr, "Error opening file.\n");
		exit(0);
	}

	std::vector<std::string> testData;
	std::vector<std::string> testLabels;

	int numTestSamples = 0;

	if (readTextFile(testFile, testData, testLabels, &numTestSamples) != 0) {
		fprintf(stderr, "Error reading file.\n");
		exit(0);
	}

	test(testData, testLabels, numTestSamples, letterPairs, priors, likelihood, C);

	std::string testText = "to these the number of pairs with space followed by a letter and the number of pairs of a letter followed by a space and the special case of two consecutive spaces";

	int prediction = classifyBayes(testText, priors, likelihood, letterPairs);

	std::cout << "TEST: " <<testText<<"\nPredicted Label: " << (prediction == 0 ? "en" : "nl") << std::endl;

	fclose(file);

	std::cout << "Most popular english pairs: ";

	int popularity = 0;
	for (int i = 0; i < likelihood.cols; i++)
	{
		if (likelihood(0, i) > popularity)
		{
			popularity = likelihood(0, i);
		}
	}

	for (int i = 0; i < likelihood.cols; i++)
	{
		if (likelihood(0, i) == popularity)
		{
			std::cout << letterPairs[i] << ", ";
		}
	}

	std::cout << "\nMost popular dutch pairs: ";
	popularity = 0;
	for (int i = 0; i < likelihood.cols; i++)
	{
		if (likelihood(1, i) > popularity)
		{
			popularity = likelihood(0, i);
		}
	}

	for (int i = 0; i < likelihood.cols; i++)
	{
		if (likelihood(1, i) == popularity)
		{
			std::cout << letterPairs[i] << ", ";
		}
	}


	Mat_<uchar> displayLikelihood(likelihood.rows, likelihood.cols);

	normalize(likelihood, displayLikelihood, 0, 255, NORM_MINMAX, CV_8U);

	namedWindow("Display Likelihood", WINDOW_NORMAL);
	resizeWindow("Display Likelihood", displayLikelihood.cols, displayLikelihood.rows * 5);

	imshow("Display Likelihood", displayLikelihood);
	waitKey(0);

}

int main() {
	project();
	return 0;
}