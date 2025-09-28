# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VTO model using the genai client."""

import uuid
from google import genai
from google.genai.types import RecontextImageSource, ProductImage, Image, RecontextImageConfig

from common.storage import store_to_gcs
from common.utils import https_url_to_gcs_uri
from config.default import Default

cfg = Default()


def generate_vto_image(
    person_gcs_url: str, product_gcs_url: str, sample_count: int, base_steps: int
) -> list[str]:
    """Generates a VTO image using the genai client's recontext_image method."""
    print(f"--- VTO GenAI Request ---")
    print(f"Person Image: {person_gcs_url}")
    print(f"Product Image: {product_gcs_url}")
    print(f"Model: {cfg.VTO_MODEL_ID}")
    print(f"-------------------------")

    client = genai.Client()

    # Convert the HTTPS URLs from the state back to GCS URIs for the API
    person_gcs_uri = https_url_to_gcs_uri(person_gcs_url)
    product_gcs_uri = https_url_to_gcs_uri(product_gcs_url)

    response = client.models.recontext_image(
        model=cfg.VTO_MODEL_ID,
        source=RecontextImageSource(
            person_image=Image.from_file(location=person_gcs_uri),
            product_images=[
                ProductImage(product_image=Image.from_file(location=product_gcs_uri))
            ],
        ),
        config=RecontextImageConfig(
            base_steps=base_steps,
            number_of_images=sample_count,
        ),
    )

    print("--- VTO GenAI Response ---")
    print(response)
    print("------------------------")

    gcs_uris = []
    if not response.generated_images:
        raise ValueError("VTO API returned no generated images.")

    for i, generated_image in enumerate(response.generated_images):
        # The API returns image bytes, so we must save them to GCS
        unique_id = uuid.uuid4()
        gcs_uri = store_to_gcs(
            folder="vto_results",
            file_name=f"vto_result_{unique_id}-{i}_.png",
            mime_type="image/png",
            contents=generated_image.data,
            decode=False,
        )
        gcs_uris.append(gcs_uri)

    return gcs_uris


from types import SimpleNamespace

def call_virtual_try_on(
    person_image_uri=None,
    product_image_uri=None,
    sample_count=1,
) -> SimpleNamespace:
    """Re-implementation of call_virtual_try_on using the genai client.
    Returns a mock response object to maintain compatibility with shop_the_look.
    """
    # This function now wraps the new generate_vto_image logic
    # and formats the output to be compatible with the old call site.
    gcs_uris = generate_vto_image(
        person_gcs_url=person_image_uri, # Note: The new function expects HTTPS URLs but we pass GCS URIs here.
        product_gcs_url=product_image_uri, # This will be handled by the https_url_to_gcs_uri inside generate_vto_image.
        sample_count=sample_count,
        base_steps=32, # Default base steps
    )

    # Mock the structure of the old PredictResponse object that shop_the_look expects.
    predictions = [{"gcsUri": uri} for uri in gcs_uris]
    mock_response = SimpleNamespace(predictions=predictions)

    return mock_response
