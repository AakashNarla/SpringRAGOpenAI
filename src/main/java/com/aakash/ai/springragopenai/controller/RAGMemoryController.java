package com.aakash.ai.springragopenai.controller;


import com.aakash.ai.springragopenai.service.RAGService;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class RAGMemoryController {

    private final ChatClient chatClient;
    private final RAGService ragService;

    public RAGMemoryController(ChatClient.Builder chatBuilder, ChatMemory chatMemory, RAGService ragService)
    {
        var memoryAdvisors = MessageChatMemoryAdvisor.builder(chatMemory).build();
        this.chatClient = chatBuilder.defaultAdvisors(memoryAdvisors).build();
        this.ragService = ragService;
    }

    @GetMapping("/askMemory")
    public String askWithMemory(@RequestParam String cid, @RequestParam String question)
    {
        var userMessage = this.ragService.generateAugmentedPrompt(question);
        var prompt = chatClient.prompt().user(userMessage);
        var advisedPrompt = prompt.advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, cid));
        var generatedContext = advisedPrompt.call().content();
        System.out.println("Generated Context: " + generatedContext);
        return generatedContext;
    }
}
