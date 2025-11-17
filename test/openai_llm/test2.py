from agent_template import OpenAILLM

if __name__ == "__main__":
    llm = OpenAILLM(model="gpt-4.1-mini", temperature=0.0)
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of France?"
    response = llm.simple_use(system_prompt=system_prompt, user_prompt=user_prompt)
    print("Response:", response)
    print(llm.get_total_fee())
