Install jNetPcap Locally 

For Windows
```bash
cd jnetpcap/win/jnetpcap-1.4.r1425
mvn install:install-file -Dfile=jnetpcap.jar -DgroupId=org.jnetpcap -DartifactId=jnetpcap -Dversion=1.4.1 -Dpackaging=jar
```

For Linux

```bash
cd jnetpcap/linux/jnetpcap-1.4.r1425
sudo mvn install:install-file -Dfile=jnetpcap.jar -DgroupId=org.jnetpcap -DartifactId=jnetpcap -Dversion=1.4.1 -Dpackaging=jar
```

if mvn is not found use 


```bash
sudo apt install maven -y
```


This adds the jNetPcap jar into your local Maven repository.

CICFlowMeter is a Java project that depends on jNetPcap — a Java wrapper for the native libpcap library (used to read .pcap files).

jNetPcap is not available online for Maven to download.
So we manually tell Maven: “Here is the jNetPcap file, use it!”

Maven is a tool for Java projects. Think of it like a package manager (like apt on Linux or npm for Node.js) but for Java.




# Build CICFlowMeter

There are two ways:


Option 1: Using Gradle (recommended)

From the project root:


```bash
# Windows
gradlew build

# Linux
./gradlew build
```


Option 2: Using Maven

```bash
mvn package
```


✅ After building:

Gradle output → CICFlowMeter/build/distributions/
Maven output → CICFlowMeter/target/

# Create the Runnable JAR

If you used Gradle:


The resulting zip file will be here:

```bash
CICFlowMeter/build/distributions/CICFlowMeter.zip
```

Unzip it — it will contain the runnable JAR.


# Run CICFlowMeter

There are two main ways to run it.


Option 1: Using Gradle Execute (direct run)


Windows

```bash
gradlew execute
```

Linux (with sudo)

```bash
sudo ./gradlew execute
```


Option 2: Run the JAR manually


Once built, run:

java -jar CICFlowMeter.jar <input_dir> <output_dir>