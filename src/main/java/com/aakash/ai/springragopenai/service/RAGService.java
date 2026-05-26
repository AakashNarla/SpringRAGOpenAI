package com.aakash.ai.springragopenai.service;

import jakarta.annotation.PostConstruct;
import org.jspecify.annotations.NonNull;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.UrlResource;
import org.springframework.stereotype.Service;

import java.net.URI;

@Service
public class RAGService {

    private final VectorStore vectorStore;

    public RAGService(EmbeddingModel embeddingModel)
    {
        this.vectorStore = SimpleVectorStore.builder(embeddingModel)
                .build();
    }

    @PostConstruct
    public void initConstruct() throws Exception
    {
        var pdfUrl = "https://certificationexams.pro/docs/pickeringisspringfield.pdf";
        var resource = new UrlResource(new URI(pdfUrl));
        // Load The PDF.
        var pages = new PagePdfDocumentReader(resource).get();
        // Add to Vector Store.
        vectorStore.add(pages);
    }

    /**
     * Generates an augmented prompt by retrieving relevant pages from the vector store based on the question.
     *
     * @param question - String
     * @return Augmented Prompt
     */
    public @NonNull String generateAugmentedPrompt(final String question) {
        var retrieveQuery = SearchRequest.builder().query(question).topK(5).build();
        var retrievedPages = vectorStore.similaritySearch(retrieveQuery);
        var augmentedContext = new StringBuilder();
        for(int i = 0; i < retrievedPages.size(); i++)
        {
            augmentedContext.append("Page ").append(i+1).append(": ").append(retrievedPages.get(i).getText()).append("\n");
        }

        var prompt = """
                You are answering questions using only the context provided form the popular Pickering is Springfield book.
                If the answer is not in the context, say "I don't know, given the pages of the book I've read.
                Maybe ask me a different question?"
                
                CONTEXT:
                %s
                
                QUESTION:
                %s
                """.formatted(augmentedContext, question);

        System.out.println(prompt);
        return prompt;
    }
}
