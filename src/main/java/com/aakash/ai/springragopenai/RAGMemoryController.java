package com.aakash.ai.springragopenai;


import jakarta.annotation.PostConstruct;
import org.jspecify.annotations.NonNull;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.UserMessage;
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
public class RAGMemoryController {

    private final ChatClient chatClient;
    private final VectorStore vectorStore;

    public RAGMemoryController(ChatClient.Builder chatBuilder, EmbeddingModel embeddingModel, ChatMemory chatMemory)
    {
        var memoryAdvisors = MessageChatMemoryAdvisor.builder(chatMemory).build();
        this.chatClient = chatBuilder.defaultAdvisors(memoryAdvisors).build();
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


    @GetMapping("/askMemory")
    public String askWithMemory(@RequestParam String cid, @RequestParam String question)
    {
        var userMessage = generateAugmentedPrompt(question);
        var prompt = chatClient.prompt().user(userMessage);
        var advisedPrompt = prompt.advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, cid));
        var generatedContext = advisedPrompt.call().content();
        System.out.println("Generated Context: " + generatedContext);
        return generatedContext;
    }

    /**
     * Generates an augmented prompt by retrieving relevant pages from the vector store based on the question.
     *
     * @param question - String
     * @return Augmented Prompt
     */
    private @NonNull String generateAugmentedPrompt(final String question) {
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
        return prompt;
    }

}
