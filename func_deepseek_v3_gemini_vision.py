"""
title: Deepseek V3/R1 Manifold Pipe with Gemini Vision Support
authors: [MCode-Team, Ethan Copping]
author_url: [https://github.com/MCode-Team, https://github.com/CoppingEthan]
funding_url: https://github.com/open-webui
version: 0.1.5
required_open_webui_version: 0.5.0
license: MIT
environment_variables:
    - DEEPSEEK_API_KEY (required)
    - GOOGLE_API_KEY (required for image processing)

User: [Text + Image]
System:
1. Gemini reads the image and generates a description.  
2. Combines the image description with the text.  
3. Sends the combined content to DeepSeek for processing.  
4. DeepSeek responds back.

# Acknowledgments
Adapted code from [Ethan Copping] to add realtime preview of the thinking process for Deepseek R1

"""

import os
import json
import time
import logging
import requests
import aiohttp
import re
import google.generativeai as genai
from typing import (
    List,
    Union,
    Generator,
    Iterator,
    Dict,
    Optional,
    AsyncIterator,
    Tuple,
)
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class CacheEntry:
    def __init__(self, description: str):
        self.description = description
        self.timestamp = time.time()


class Pipe:
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
    TOTAL_MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB total
    REQUEST_TIMEOUT = (3.05, 60)
    CACHE_EXPIRATION = 30 * 60  # 30 minutes in seconds
    MODEL_MAX_TOKENS = {
        "deepseek-chat": 8192,
        "deepseek-reasoner": 8192,
    }

    class Valves(BaseModel):
        DEEPSEEK_BASE_URL: str = Field(
            default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            description="Your DeepSeek Base URL",
        )
        DEEPSEEK_API_KEY: str = Field(
            default=os.getenv("DEEPSEEK_API_KEY", ""),
            description="Your DeepSeek API key",
        )
        GOOGLE_API_KEY: str = Field(
            default=os.getenv("GOOGLE_API_KEY", ""),
            description="Your Google API key for image processing",
        )
        THINK_XML_TAG: str = Field(
            default=os.getenv("THINK_XML_TAG", "thinking"),
            description="XML tag used for thinking content",
        )

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "deepseek"
        self.name = "deepseek/"
        self.valves = self.Valves()
        self.request_id = None
        self.image_cache = {}

    @staticmethod
    def get_model_id(model_name: str) -> str:
        """Extract just the base model name from any format"""
        return model_name.replace(".", "/").split("/")[-1]

    def get_deepseek_models(self) -> List[Dict[str, str]]:
        """Fetch available models from Deepseek API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }
            response = requests.get(
                f"{self.valves.DEEPSEEK_BASE_URL}/models", headers=headers, timeout=10
            )
            response.raise_for_status()
            models_data = response.json()
            return [
                {"id": model["id"], "name": model["id"]}
                for model in models_data.get("data", [])
            ]
        except Exception as e:
            if DEBUG:
                print(f"Error getting models: {e}")
            return []

    def pipes(self) -> List[dict]:
        return self.get_deepseek_models()

    def clean_expired_cache(self):
        """Remove expired entries from cache"""
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self.image_cache.items()
            if current_time - entry.timestamp > self.CACHE_EXPIRATION
        ]
        for key in expired_keys:
            del self.image_cache[key]

    def extract_images_and_text(self, message: Dict) -> Tuple[List[Dict], str]:
        """Extract images and text from a message."""
        images = []
        text_parts = []
        content = message.get("content", "")

        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    images.append(item)
        else:
            text_parts.append(content)

        return images, " ".join(text_parts)

    async def process_image_with_gemini(
        self, image_data: Dict, __event_emitter__=None
    ) -> str:
        """Process a single image with Gemini and return its description."""
        try:
            if not self.valves.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for image processing")

            # Clean expired cache entries
            self.clean_expired_cache()

            # Create cache key
            image_url = image_data.get("image_url", {}).get("url", "")
            image_key = image_url
            if image_url.startswith("data:image"):
                image_key = image_url.split(",", 1)[1] if "," in image_url else ""

            # Check cache
            if image_key in self.image_cache:
                logging.info(f"Using cached image description for {image_key[:30]}...")
                return self.image_cache[image_key].description

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Processing new image...",
                            "done": False,
                        },
                    }
                )

            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")

            if image_url.startswith("data:image"):
                image_data = image_url.split(",", 1)[1] if "," in image_url else ""
                image_part = {
                    "inline_data": {"mime_type": "image/jpeg", "data": image_data}
                }
            else:
                image_part = {"image_url": image_url}

            prompt = "Give a clear and detailed description of this image."
            response = model.generate_content([prompt, image_part])
            description = response.text

            # Cache the result
            self.image_cache[image_key] = CacheEntry(description)

            # Limit cache size (keep 100 most recent)
            if len(self.image_cache) > 100:
                oldest_key = min(
                    self.image_cache.keys(), key=lambda k: self.image_cache[k].timestamp
                )
                del self.image_cache[oldest_key]

            return description

        except Exception as e:
            logging.error(f"Error processing image with Gemini: {str(e)}")
            return f"[Error processing image: {str(e)}]"

    async def process_messages(
        self, messages: List[Dict], __event_emitter__=None
    ) -> List[Dict]:
        """Process messages, replacing images with their descriptions."""
        processed_messages = []

        for message in messages:
            images, text = self.extract_images_and_text(message)

            if images:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Found {len(images)} image(s) to process",
                                "done": False,
                            },
                        }
                    )

                # Process each image and get descriptions
                image_descriptions = []
                for idx, image in enumerate(images, 1):
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Processing image {idx} of {len(images)}",
                                    "done": False,
                                },
                            }
                        )
                    description = await self.process_image_with_gemini(
                        image, __event_emitter__
                    )
                    image_descriptions.append(f"[Image Description: {description}]")

                # Combine original text with image descriptions
                combined_content = text + " " + " ".join(image_descriptions)
                processed_messages.append(
                    {"role": message["role"], "content": combined_content.strip()}
                )

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "All images processed successfully",
                                "done": False,
                            },
                        }
                    )
            else:
                processed_messages.append(message)

        return processed_messages

    def format_thinking_tags(self, text: str) -> str:
        """Format content within thinking XML tags into markdown blockquotes."""
        pattern = r"<{}>(.*?)</{}>".format(
            re.escape(self.valves.THINK_XML_TAG), re.escape(self.valves.THINK_XML_TAG)
        )
        regex = re.compile(pattern, flags=re.DOTALL)

        def replacer(match):
            thinking_content = match.group(1).strip()
            formatted_lines = [f"> {line}" for line in thinking_content.splitlines()]
            return "\n".join(formatted_lines)

        formatted_text = regex.sub(replacer, text)
        return formatted_text

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[str, Generator, Iterator]:
        if not self.valves.DEEPSEEK_API_KEY:
            error_msg = "Error: DEEPSEEK_API_KEY is required"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"content": error_msg, "format": "text"}

        try:
            system_message, messages = pop_system_message(body.get("messages", []))
            processed_messages = await self.process_messages(
                messages, __event_emitter__
            )
            if "model" not in body:
                raise ValueError("Model name is required")

            model_id = self.get_model_id(body["model"])
            max_tokens_limit = self.MODEL_MAX_TOKENS.get(model_id, 8192)

            if system_message:
                processed_messages.insert(
                    0, {"role": "system", "content": str(system_message)}
                )

            payload = {
                "model": model_id,
                "messages": processed_messages,
                "max_tokens": min(
                    body.get("max_tokens", max_tokens_limit), max_tokens_limit
                ),
                "temperature": float(body.get("temperature", 0.7)),
                "top_k": (
                    int(body.get("top_k", 40))
                    if body.get("top_k") is not None
                    else None
                ),
                "top_p": (
                    float(body.get("top_p", 0.9))
                    if body.get("top_p") is not None
                    else None
                ),
                "stream": body.get("stream", False),
            }
            payload = {k: v for k, v in payload.items() if v is not None}

            headers = {
                "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }

            if __event_emitter__ and model_id == "deepseek-reasoner":
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "thinking...", "done": False},
                    }
                )

            if payload["stream"]:
                return self._stream_response(
                    url=f"{self.valves.DEEPSEEK_BASE_URL}/chat/completions",
                    headers=headers,
                    payload=payload,
                    __event_emitter__=__event_emitter__,
                    model_id=model_id,
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.valves.DEEPSEEK_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_msg = (
                            f"Error: HTTP {response.status}: {await response.text()}"
                        )
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": error_msg,
                                        "done": True,
                                    },
                                }
                            )
                        return {"content": error_msg, "format": "text"}

                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        message = result["choices"][0]["message"]
                        content = message.get("content") or ""
                        reasoning_content = message.get("reasoning_content") or ""
                        combined_content = ""
                        if reasoning_content:
                            combined_content += f"<{self.valves.THINK_XML_TAG}>\n{reasoning_content.strip()}\n</{self.valves.THINK_XML_TAG}>\n\n"
                        combined_content += content
                        final_response = self.format_thinking_tags(combined_content)
                        if __event_emitter__ and model_id == "deepseek-reasoner":
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": "Request completed successfully",
                                        "done": True,
                                    },
                                }
                            )
                        return final_response
                    return ""

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"content": error_msg, "format": "text"}

    async def _stream_response(
        self,
        url: str,
        headers: dict,
        payload: dict,
        __event_emitter__=None,
        model_id: str = "",
    ) -> AsyncIterator[str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if __event_emitter__ and model_id == "deepseek-reasoner":
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": "thinking...", "done": False},
                            }
                        )
                    if response.status != 200:
                        error_msg = (
                            f"Error: HTTP {response.status}: {await response.text()}"
                        )
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"description": error_msg, "done": True},
                                }
                            )
                        yield error_msg
                        return

                    reasoning_content = ""
                    content = ""
                    is_thinking = False
                    last_status_update = time.time()
                    status_dots = 0

                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})

                                    # Handle reasoning content (thinking phase)
                                    if (
                                        "reasoning_content" in delta
                                        and delta["reasoning_content"]
                                    ):
                                        if not is_thinking:
                                            is_thinking = True
                                            yield "> "  # Start blockquote only once

                                        delta_text = delta["reasoning_content"]
                                        # Handle new lines in thinking content
                                        if "\n" in delta_text:
                                            delta_text = delta_text.replace(
                                                "\n", "\n> "
                                            )
                                        yield delta_text
                                        reasoning_content += delta_text

                                    # Handle final response content
                                    if "content" in delta and delta["content"]:
                                        if is_thinking:
                                            is_thinking = False
                                            yield "\n\n"  # Add separation between thinking and response

                                        content_chunk = delta["content"]
                                        content += content_chunk
                                        yield content_chunk

                                    if (
                                        model_id == "deepseek-reasoner"
                                        and __event_emitter__
                                    ):
                                        current_time = time.time()
                                        if current_time - last_status_update > 1:
                                            status_dots = (status_dots % 3) + 1
                                            await __event_emitter__(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": f"thinking{'...'[:status_dots]}",
                                                        "done": False,
                                                    },
                                                }
                                            )
                                            last_status_update = current_time

                                    if (
                                        data["choices"][0].get("finish_reason")
                                        == "stop"
                                    ):
                                        if (
                                            __event_emitter__
                                            and model_id == "deepseek-reasoner"
                                        ):
                                            await __event_emitter__(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": "Request completed",
                                                        "done": True,
                                                    },
                                                }
                                            )
                                        break

                            except json.JSONDecodeError as e:
                                logging.error(
                                    f"Failed to parse streaming response: {e}"
                                )
                                continue

        except Exception as e:
            error_msg = f"Stream error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            yield error_msg
