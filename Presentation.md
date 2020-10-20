## Motivation: 
WE technique, represent word in vector form.

What we expect in inputs of ML is numerical vectors instead of string sequences, so WE can provide us with more probabilties to 
		transform sentences.

## Advantages:
1. Word Similarities: powerful, Euclidean Distance

	*Why powerful*: Instead of just setting 1 or 0 in each position of vector, representing them with float number can capture more info about  a word.
2. Scalibility: In WE, we choose a vector with a dimension and we can freely choose what the dimension is depending on the computing resourses we have and the sepcific task we deal with.
## Implementations: ML(Neural Network), 2 main streams
1. W2V: Based on context, we predict a word, then we compare them with the actual word appeared, and update the NN.
	* CBOW:
        1. We have a sentence, 'I drink coffee everyday'.
		2. We use the model to predict a bunch of words' possibilities when we have 'I drink everyday', 
		3. and we know that the actual word is 'coffee', than we can calculate the loss of the probabilties sequences to the actual word.
		4. Use the loss to BP to the whole NN.
	* SG:
		1. The basic structure of model in SG is like what we have in CBOW.
		2. The only difference: We use the model predict context "I drink everyday" when we have a word 'coffee'.
2. GloVe: Based on context, we get the statistical info of co-occurences of words.
	1. We build a matrix X, elements X_{ij} tabulate the number of times word j occurs in the context of word i.
	2. Then we use each eles in X as train data to update our model.
3. Swivel: improved version of GloVe(2 improves)
	1. Parrallel it
	2. Some words' co-occurence frequency is too Low: use more smooth loss func.

