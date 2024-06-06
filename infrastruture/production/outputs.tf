output "vm_ssh_connection_string" {
  description = "SSH connection string to the Azure virtual machine"
  value       = "ssh ${var.admin_username}@${module.azure_vm_setup.public_ip_address}"
}
