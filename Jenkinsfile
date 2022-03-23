#!/usr/bin/env groovy

node {
	stage('Checkout') {
		checkout(scm)
	}

	stage('Docker build') {
		sh('make build')
	}

	stage('Docker check') {
		sh('make check')
	}

	stage('Parsing version') {
		// Get the module version
		TRUE_VERSION = sh(script: 'make module_version', returnStdout: true).trim()
		VERSION = sh(script: "echo ${TRUE_VERSION} | sed 's/-.*\$//'", returnStdout: true).trim()
		BRANCH_VERSION = sh(
			script: "echo ${TRUE_VERSION} | grep -Eo '^[0-9]+(\\.[0-9]+)?'", returnStdout: true
			).trim()

		// Log parsed version
		echo("TRUE_VERSION = ${TRUE_VERSION}")
		echo("VERSION = ${VERSION}")
		echo("BRANCH_VERSION = ${BRANCH_VERSION}")
	}

	stage('Docker push') {
		// Get the module name
		module_name = sh(script: 'make module_name', returnStdout: true).trim()

		docker.withRegistry(
			"https://${env.AWS_ACCOUNT_ID}.dkr.ecr.eu-central-1.amazonaws.com",
			"ecr:eu-central-1:${env.ENVIRONMENT}"
			) {
			// Push the version tags
			image = docker.image(module_name)
			image.push(TRUE_VERSION)
			image.push(VERSION)
			image.push(BRANCH_VERSION)
			image.push('latest')
		}
	}
}
