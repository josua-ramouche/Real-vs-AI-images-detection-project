FROM ubuntu
RUN apt update -y && apt install -y openjdk-17-jdk
RUN apt install -y nano
RUN apt -y install net-tools iputils-ping
WORKDIR /app
COPY ./Documents/PredictionMS-0.0.1-SNAPSHOT.jar  /app/PredictionMS.jar
EXPOSE 50001
CMD ["java", "-jar", "/app/PredictionMS.jar"]
CMD ["/bin/bash"]

