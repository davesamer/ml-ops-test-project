variable "admin_username" {
  description = "Username for the virtual machine"
  type        = string
}

variable "admin_password" {
  description = "Password for the virtual machine"
  type        = string
  sensitive   = true
}
