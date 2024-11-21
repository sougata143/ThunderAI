provider "aws" {
  region = "us-west-2"
}

resource "aws_ecs_cluster" "thunderai_cluster" {
  name = "thunderai-cluster"
}

resource "aws_ecs_task_definition" "thunderai_task" {
  family                   = "thunderai-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"

  container_definitions = jsonencode([
    {
      name      = "thunderai-container"
      image     = "your-ecr-repo/thunderai:latest"
      cpu       = 512
      memory    = 1024
      essential = true
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
        }
      ]
    }
  ])
}

resource "aws_ecs_service" "thunderai_service" {
  name            = "thunderai-service"
  cluster         = aws_ecs_cluster.thunderai_cluster.id
  task_definition = aws_ecs_task_definition.thunderai_task.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = ["subnet-12345678"]
    security_groups = ["sg-12345678"]
  }
} 