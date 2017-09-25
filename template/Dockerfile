From tensorflow/tensorflow:latest-py3

# Install necessary packages
RUN apt update
RUN apt install gcc make libc-dev g++ bzip2 libssl-dev build-essential -y

# Hop into /template
WORKDIR /template

# Install Java 8 (Neo4j dependency)
RUN add-apt-repository ppa:webupd8team/java
RUN apt-get update
RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections
RUN apt-get install oracle-java8-installer -y

# Install and start Neo4j (disabling auth, enabling host machine access)
RUN wget -O - http://debian.neo4j.org/neotechnology.gpg.key | apt-key add - && \
    echo 'deb http://debian.neo4j.org/repo stable/' \
      | tee /etc/apt/sources.list.d/neo4j.list && \
    apt-get update && \
    apt-get install --no-install-recommends -y neo4j && \
    sed -i "s|#dbms.security.auth_enabled=false|dbms.security.auth_enabled=false|g" /etc/neo4j/neo4j.conf && \
    sed -i "s|#dbms.connector.bolt.listen_address=:7687|dbms.connector.bolt.listen_address=0.0.0.0:7687|g" /etc/neo4j/neo4j.conf && \
    sed -i "s|#dbms.connector.http.listen_address=:7474|dbms.connector.http.listen_address=0.0.0.0:7474|g" /etc/neo4j/neo4j.conf && \
    sed -i "s|#dbms.connector.https.listen_address=:7473|dbms.connector.https.listen_address=0.0.0.0:7473|g" /etc/neo4j/neo4j.conf
RUN service neo4j start

# Install Python requirements
COPY ./template/requirements.txt ./requirements.txt
RUN export C_INCLUDE_PATH=/usr/include
RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt

