## LiteLLM Cache

LiteLLM Cache is a proxy server designed to cache your LLM requests, helping to reduce costs and improve efficiency.

### Requirements
- Docker Compose
- Docker

### Setup Instructions

1. **Configure Settings:**
   - Navigate to `./config.yaml` and update the configuration as per your requirements. For more information, visit [LiteLLM Documentation](https://litellm.vercel.app/).

2. **Prepare Environment Variables:**
   - Create a `.env` file from the `.env.sample` file. Adjust the details in `.env` to match your `config.yaml` settings.

3. **Start the Docker Container:**
   ```bash
   docker-compose up -d
   ```

4. **Update Your LLM Server URL:**
   - Change the LLM calling server URL in your application to `http://0.0.0.0:4000`.
   
   For example, using the OpenAI Python SDK:
   ```python
   from openai import OpenAI

   llm = OpenAI(
       base_url='http://0.0.0.0:4000'
   )
   ```

With these steps, your LLM requests will be routed through the LiteLLM Cache proxy server, optimizing performance and reducing costs.