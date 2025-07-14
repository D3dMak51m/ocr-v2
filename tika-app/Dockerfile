# Base image with OpenJDK 17
FROM openjdk:17-jdk-slim

RUN apt-get update && \
    apt-get install -y iputils-ping wget && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for Tika version (optional, for clarity)
ENV TIKA_VERSION=standard-3.0.0-BETA2.jar

# Copy the already downloaded Tika server JAR file from your local machine to the container
COPY tika-server-$TIKA_VERSION /tika-server.jar

# Expose port 9998 where Tika server runs
EXPOSE 9998

# Run the Tika server, binding to all IP addresses (0.0.0.0)
CMD ["java", "-jar", "/tika-server.jar", "-p", "9998", "-host", "0.0.0.0"]
