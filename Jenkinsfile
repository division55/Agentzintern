
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
    }

    post {
        success {
            echo 'Build and syntax checks passed!'
        }

        failure {
            echo 'Pipeline failed. Check the stage logs.'
        }
    }
}
