import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;

/**
 * Created by user on 2016-02-28.
 */
public class WordVec {

    public static String name = "BernieSanders";

    public static void main(String[] args){


        /*try {
            WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File("words.txt"));
        }catch(Exception e){
        }*/
        Word2Vec vec = readInData();


        double sim = vec.similarity("people", "money");
        System.out.println("Similarity : " + sim);
        Collection<String> similar = vec.wordsNearest("day", 10);
        System.out.println("Words close to day : " + similar);

        //output: [night, week, year, game, season, during, office, until, -]

        double ClintonTrump = vec.similarity("donaldtrump", "hillaryclinton");
        double ClintonSanders = vec.similarity("berniesanders", "hillaryclinton");
        double SandersTrump = vec.similarity("donaldtrump", "berniesanders");
        double ClintonTrumpTrunc = vec.similarity("trump", "clinton");
        double ClintonSandersTrunc = vec.similarity("sanders", "clinton");
        double SandersTrumpTrunc = vec.similarity("trump", "sanders");

        System.out.println("Similarity between Clinton and Trump : " + ClintonTrump);
        System.out.println("Similarity between Clinton and Sanders : " + ClintonSanders );
        System.out.println("Similarity between Sanders and Trump : " + SandersTrump);

        System.out.println("Similarity between Clinton and Trump Trunc : " + ClintonTrumpTrunc);
        System.out.println("Similarity between Clinton and Sanders Trunc: " + ClintonSandersTrunc );
        System.out.println("Similarity between Sanders and Trump Trunc: " + SandersTrumpTrunc);

        Collection<String> ClintonSimilar = vec.wordsNearest("hillaryclinton", 25);
        Collection<String> SandersSimilar = vec.wordsNearest("berniesanders", 25);
        Collection<String> TrumpSimilar = vec.wordsNearest("donaldtrump", 25);

        Collection<String> ClintonSimilarTrunc = vec.wordsNearest("clinton", 25);
        Collection<String> SandersSimilarTrunc = vec.wordsNearest("sanders", 25);
        Collection<String> TrumpSimilarTrunc = vec.wordsNearest("trump", 25);

        System.out.println("Words close to Clinton : " + ClintonSimilar);
        System.out.println("Words close to Sanders : " + SandersSimilar);
        System.out.println("Words close to Trump : " + TrumpSimilar);

        System.out.println("Words close to Clinton Trunc: " + ClintonSimilarTrunc);
        System.out.println("Words close to Sanders Trunc: " + SandersSimilarTrunc);
        System.out.println("Words close to Trump Trunc: " + TrumpSimilarTrunc);


       /* try {
            String output = String.format("%swords.txt",name);
            WordVectorSerializer.writeWordVectors(vec, output);
        }catch(Exception e){
            e.printStackTrace();
        }*/
    }

    private static void plotTable(Word2Vec vec) {
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(1000)
                .stopLyingIteration(250)
                .learningRate(500)
                .useAdaGrad(false)
                .theta(0.5)
                .setMomentum(0.5)
                .normalize(true)
                .usePca(false)
                .build();
        vec.lookupTable().plotVocab(tsne);
    }

    private static Word2Vec readInData() {
        System.out.println("FileReading Starts");
        String path = String.format("C://Anaconda/Math76/FinalData/%sTexts2.csv", name);
        SentenceIterator iter = new LineSentenceIterator(new File(path));
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        System.out.println("Tokening Starts");

        final EndingPreProcessor preProcessor = new EndingPreProcessor();
        TokenizerFactory tokenizer = new DefaultTokenizerFactory();
        tokenizer.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                token = token.toLowerCase();
                String base = preProcessor.preProcess(token);
                base = base.replaceAll("\\d", "d");
                //if (base.endsWith("ly") || base.endsWith("ing"))
                // System.out.println();
                return base;
            }
        });

        int batchSize = 1000;
        int iterations = 3;
        int layerSize = 150;


        System.out.println("Build model....");
        Word2Vec vec = new Word2Vec.Builder()
                .batchSize(batchSize) //# words per minibatch.
                .minWordFrequency(5) //
                .useAdaGrad(false) //
                .layerSize(layerSize) // word feature vector size
                .iterations(iterations) // # iterations to train
                .learningRate(0.025) //
                .minLearningRate(1e-3) // learning rate decays wrt # words. floor learning
                .negativeSample(0) // sample size 10 words
                .iterate(iter) //
                .tokenizerFactory(tokenizer)
                .build();
        vec.fit();
        return vec;
    }
}
