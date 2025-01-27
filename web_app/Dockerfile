FROM python:3.8-slim
WORKDIR /app
ADD . /app
RUN python -m pip install --upgrade pip
RUN pip install flask
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install pillow
RUN pip install sentencepiece
RUN pip install sacremoses
RUN pip install torch
RUN pip install transformers
EXPOSE 5000
ENV NAME describe_image
CMD ["python","app.py"]

