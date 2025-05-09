
FROM python:3.10-slim


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


COPY . . 


CMD ["fastapi", "run", "src.main.py", "--port", "8000"]