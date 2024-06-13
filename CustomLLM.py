from typing import List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import requests


class EYQIncubator(LLM):
    """
    Custom LLM class for EY Incubator API
    """
    end_point: str
    x_api_key: str
    model: str
    api_version: str
    temperature: float = 0.5

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> str:
        """
        Make an API call to the Azure OpenAI instance using the specified
        prompt and return the response.
        """
        headers = {
            "x-api-key": self.x_api_key
        }
        query_params = {
            "api-version": self.api_version
        }
        body = {
            "messages": [
                {"role": "system",
                 "content": """You are a helpful AI assistant that answers question based on context.
                 """},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
        }
        full_path = self.end_point + "/openai/deployments/" + self.model + "/chat/completions"
        response = requests.post(full_path, json=body, headers=headers, params=query_params)
        status_code = response.status_code
        response = response.json()
        if status_code == 200:
            body["messages"].append({"role": "system", "content": response["choices"][0]["message"]["content"]})
        else:
            print("\nError: ", status_code)
            print("Response: ", response["error"] + "\n")

        result = response["choices"][0]["message"]["content"]
        if len(result) < 1:
            result = "Nothing Returned"

        # Return the response from the API
        return str(result)
