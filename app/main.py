import random

from fastapi import FastAPI, HTTPException, Path, Query, status
from fastapi.responses import PlainTextResponse, RedirectResponse

app = FastAPI()


@app.get("/generate")
def generate_haiku(
    prompt: str = Query(
        None,
        description="The prompt to begin the generated haiku with. If not given, a random one will be chosen.",
        max_length=50,
    ),
    seed: int = Query(
        None,
        description="Seed the RNG. If not given, a random seed will be generated, used, and returned in the JSON response for reproducibility.",
        gt=0,
        lt=2 ** 32,
    ),
    temperature: float = Query(
        1.0,
        description="The temperature to use when generating the haiku. Higher temperatures result in more randomness.",
        gt=0,
    ),
    k: int = Query(
        0,
        description="The number of highest probability vocabulary tokens to keep for top-k filtering.",
        ge=0,
    ),
    p: float = Query(
        0.9,
        description="The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.",
        ge=0,
        le=1,
    ),
    max_tokens: int = Query(
        20, description="The max length of the sequence to be generated.", gt=0,
    ),
):
    """Generate a random haiku based on the given prompt."""
    return {"seed": seed, "prompt": prompt, "haiku": ["1", "2",]}


@app.get("/generated")
async def random_generated_haiku():
    """Get a random generated haiku."""
    # TODO: Dynamically update the upper bound based on the number of generated haiku.
    # Should try to avoid file I/O for this, however.
    n = random.randrange(0, 1200)
    return RedirectResponse(f"/generated/{n}")


# TODO: Find a way to cache the generated haiku in a thread-safe manner.
@app.get("/generated/{n}", response_class=PlainTextResponse)
def generated_haiku(n: int = Path(..., description="The haiku index", ge=0)):
    """Get the nth generated haiku."""
    # TODO Dynamically update the upper bound.
    if n > 1200:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="That haiku doesn't exist!")
    return f"{n} spoonfuls / of medication / and loneliness"


@app.get("/data")
async def random_training_set_haiku():
    """Get a random human-written haiku from the training set."""
    # TODO: Pick this value at the API start time.
    n = random.randrange(0, 55000)
    return RedirectResponse(f"/data/{n}")


# TODO: Find a way to keep the training set dataframe in memory in a thread-safe manner.
@app.get("/data/{n}")
def training_set_haiku(n: int = Path(..., description="The haiku index", ge=0)):
    """Get the nth human-written haiku from the training set."""
    # TODO: Update this value at API start time.
    if n > 55000:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="That haiku doesn't exist!")
    return f"{n} a love letter / to the butterfly gods / with strategic misspellings"
