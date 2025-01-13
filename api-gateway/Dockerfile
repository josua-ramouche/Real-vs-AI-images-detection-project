FROM ubuntu
RUN apt update -y && apt install -y openjdk-17-jdk
RUN apt install -y nano
RUN apt -y install net-tools iputils-ping
WORKDIR /app
COPY ./Documents/api-gateway-0.0.1-SNAPSHOT.jar  /app/api-gateway.jar
EXPOSE 8080
CMD ["java", "-jar", "/app/api-gateway.jar"]
CMD ["/bin/bash"]

