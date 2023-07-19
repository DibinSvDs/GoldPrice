FROM python:3.10
EXPOSE 8080
WORKDIR /app
COPY . ./
RUN pip install -r requirements.txt && pip install openpyxl
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]