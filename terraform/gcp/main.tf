provider "google" {
  credentials = file("<path-to-your-service-account-key>.json")
  project     = "<your-gcp-project-id>"
  region      = "us-central1"
}

resource "google_container_cluster" "thunderai_cluster" {
  name     = "thunderai-cluster"
  location = "us-central1"

  initial_node_count = 3

  node_config {
    machine_type = "e2-medium"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}

resource "google_container_node_pool" "primary_preemptible_nodes" {
  name       = "primary-preemptible-nodes"
  location   = google_container_cluster.thunderai_cluster.location
  cluster    = google_container_cluster.thunderai_cluster.name
  node_count = 1

  node_config {
    preemptible  = true
    machine_type = "e2-medium"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}

resource "kubernetes_deployment" "thunderai" {
  metadata {
    name = "thunderai"
    labels = {
      app = "thunderai"
    }
  }

  spec {
    replicas = 2

    selector {
      match_labels = {
        app = "thunderai"
      }
    }

    template {
      metadata {
        labels = {
          app = "thunderai"
        }
      }

      spec {
        container {
          image = "gcr.io/<your-gcp-project-id>/thunderai:latest"
          name  = "thunderai"

          ports {
            container_port = 8000
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "thunderai" {
  metadata {
    name = "thunderai"
  }

  spec {
    selector = {
      app = "thunderai"
    }

    type = "LoadBalancer"

    port {
      port        = 80
      target_port = 8000
    }
  }
} 