# Base image with OpenJDK 17
FROM openjdk:17-jdk-slim

# Install required packages and fonts for better Unicode support
RUN apt-get update && \
    apt-get install -y \
    iputils-ping \
    wget \
    locales \
    fonts-dejavu-core \
    fonts-liberation \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Set up locale for UTF-8 support
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

# Set Java options for proper encoding
ENV JAVA_OPTS="-Dfile.encoding=UTF-8 -Dsun.jnu.encoding=UTF-8"

# Set environment variables for Tika version
ENV TIKA_VERSION=standard-3.0.0-BETA2.jar

# Copy the Tika server JAR file
COPY tika-server-$TIKA_VERSION /tika-server.jar

# Expose port 9998 where Tika server runs
EXPOSE 9998

# Run the Tika server with UTF-8 encoding support
CMD ["sh", "-c", "java $JAVA_OPTS -jar /tika-server.jar -p 9998 -host 0.0.0.0"]