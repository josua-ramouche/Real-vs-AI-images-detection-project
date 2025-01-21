FROM ubuntu
RUN apt update -y && apt install -y openjdk-17-jdk
RUN apt -y install net-tools iputils-ping
WORKDIR /app
COPY ./target/DiscoveryMS-0.0.1-SNAPSHOT.jar  /app/DiscoveryMS.jar
EXPOSE 8761
CMD ["java", "-jar", "/app/DiscoveryMS.jar"]

