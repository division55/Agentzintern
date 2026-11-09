
pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        IMAGE_NAME = 'agentzintern'
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
                echo 'Source code checked out successfully.'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    env.IMAGE_TAG = "${env.IMAGE_NAME}:${env.BUILD_NUMBER}"
                }

                sh 'docker build -t $IMAGE_TAG .'
            }
        }

        stage('Python Syntax Check') {
            steps {
                sh '''
                    docker run --rm $IMAGE_TAG \
                    python -m compileall -q agent.py rag.py ingest.py
                '''
            }
        }

        stage('SAST - Bandit') {
            steps {
                sh '''
                    docker run --rm \
                    -v "$WORKSPACE:/src" \
                    -w /src \
                    python:3.13-slim \
                    sh -c "pip install --no-cache-dir bandit && bandit -r agent.py rag.py ingest.py --exit-zero"
                '''
            }
        }

        stage('SCA - pip-audit') {
            steps {
                sh '''
                    docker run --rm \
                    -v "$WORKSPACE:/src" \
                    -w /src \
                    python:3.13-slim \
                    sh -c "pip install --no-cache-dir pip-audit && pip-audit -r requirements.txt"
                '''
            }
        }

        stage('Secret Scanning - Gitleaks') {
            steps {
                sh '''
                    docker run --rm \
                    -v "$WORKSPACE:/src" \
                    zricethezav/gitleaks:latest \
                    detect --source=/src --no-git --redact --exit-code 1
                '''
            }
        }

        stage('Filesystem Security Scan - Trivy') {
            steps {
                sh '''
                    docker run --rm \
                    -v "$WORKSPACE:/src" \
                    aquasec/trivy:latest \
                    fs --scanners vuln,misconfig,secret \
                    --severity HIGH,CRITICAL \
                    --exit-code 1 /src
                '''
            }
        }
    }

    post {
        success {
            echo 'All pipeline stages completed successfully!'
        }

        failure {
            echo 'Pipeline failed. Check the security scan and build logs.'
        }

        always {
            echo 'Agentzintern DevSecOps pipeline execution completed.'
        }
    }
}
