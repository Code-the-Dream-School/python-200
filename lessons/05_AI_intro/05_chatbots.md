# Chatbot

A chatbot is a computer program designed to simulate human conversation with an end user. Chatbots can assist with tasks, answer questions, or generate content, providing an interactive and responsive experience.

## AI Chatbots

Modern AI chatbots go beyond pre-programmed FAQs. By leveraging generative AI and an organization’s knowledge base, chatbots can automatically generate answers to a wide range of questions.

**Conversational AI chatbots**: Understand user questions or comments and respond in a human-like manner.

**Generative AI chatbots**: Create new content as output, such as personalized messages, recommendations, or summaries.

### Typical Use Cases

- Providing timely, always-on customer service or HR assistance
- Offering personalized recommendations in e-commerce
- Promoting products and services via chatbot marketing
- Defining fields within forms or financial applications
- Scheduling appointments and intake for healthcare offices
- Sending automated reminders for time- or location-based tasks

### Benefits

- Improve customer engagement
- Reduce operational costs

### Limitations

Accuracy risk: Chatbots may produce inaccurate or irrelevant answers (“hallucinations”), requiring escalation to human agents.

Security risk: Sensitive information could inadvertently become part of the model’s data, potentially violating privacy and security policies.

### Example: Kudos Chatbot

This chatbot helps employees generate short, positive, and specific recognition messages for their colleagues.

Prompt Example:

You are a friendly and professional assistant that helps employees write kudos messages. Highlight achievements, teamwork, or personal qualities. If the user cannot recall details, ask guiding questions to help them remember. Always keep the tone warm, genuine, and encouraging.

Sample code and output:
content: user request
```
completion = client.chat.completions.create(model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": "You are an assistant, helps employees write kudos messages. Highlight achievements, teamwork, or personal qualities."},
                            {"role": "user", "content": content}
                        ]
                    )
            suggestions = completion.choices[0].message.content
```

### Best Practices
- Define clear use cases
- Understand model limitations
- Manage context effectively
- Prioritize user privacy and security
- Implement user authentication
- Provide clear instructions

### Key Principles for Writing Prompts
- Write Clear, Specific Instructions
- Specify exactly what the task is
- Indicate how the output should be structured
- Include conditions to check
- Optionally, use few-shot prompts (provide examples of successful completion before asking the model to perform the task)
- Give Time for the Model to Reason
- Specify step-by-step instructions
- Instruct the model to work out its solution before concluding

### Building a Custom Chatbot

- Define roles to set the assistant’s behavior
- Use a system message for context
- The assistant model interfaces with users, receiving input and returning output
- Adjust temperature to control creativity; higher values produce more varied responses