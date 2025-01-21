FROM ubuntu
RUN apt update -y && apt install -y openjdk-17-jdk
RUN apt -y install net-tools iputils-ping
WORKDIR /app
COPY ./target/PredictionMS-0.0.1-SNAPSHOT.jar  /app/PredictionMS.jar
EXPOSE 8081
CMD ["java", "-jar", "/app/PredictionMS.jar"]

