provider "azurerm" {
  features {}
}

module "azure_vm_setup" {
  source               = "github.com/davesamer/terraform-modules-azure//services/azure-vm-setup"
  azure_vm_setup_name  = "project-churn-prediction"
  location             = "eastus2"
  vm_size              = "Standard_DS1_v2"
  admin_username       = var.admin_username
  admin_password       = var.admin_password
}

resource "azurerm_network_security_rule" "ssh_rule" {
  name                        = "${module.azure_vm_setup.azure_vm_setup_name}_ssh_rule"
  priority                    = 100
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_range      = "22"
  source_address_prefix       = "*"
  destination_address_prefix  = "*"
  network_security_group_name = module.azure_vm_setup.nsg_name
  resource_group_name         = module.azure_vm_setup.resource_group_name
}

resource "azurerm_network_security_rule" "port_8000_rule" {
  name                        = "${module.azure_vm_setup.azure_vm_setup_name}_port_8000_rule"
  priority                    = 101
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_range      = "8000"
  source_address_prefix       = "*"
  destination_address_prefix  = "*"
  network_security_group_name = module.azure_vm_setup.nsg_name
  resource_group_name         = module.azure_vm_setup.resource_group_name
}