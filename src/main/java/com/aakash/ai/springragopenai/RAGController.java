package com.aakash.ai.springragopenai;


import jakarta.annotation.PostConstruct;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.UrlResource;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.net.URI;

@RestController
public class RAGController {

    private ChatClient chatClient;
    VectorStore vectorStore;

    public RAGController(ChatClient.Builder chatBuilder, EmbeddingModel embeddingModel)
    {
        this.chatClient = chatBuilder.build();
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


    @GetMapping("/ask")
    public String ask(@RequestParam String question)
    {
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
        var generatedContext = chatClient.prompt(prompt).call().content();
        System.out.println("Generated Context: " + generatedContext);
        return generatedContext;

    }

}
