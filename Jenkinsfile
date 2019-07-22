pipeline {
  agent { dockerfile true }
  stages {
    stage('GPU Check') {
        steps {
            sh 'nvcc --version'
        }
    }

    stage('Build') {
        steps {
            sh 'make -j check'
        }
    }

    stage('Test') {
        steps {
            sh ''
        }
    }
  }
}