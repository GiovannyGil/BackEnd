using Microsoft.AspNetCore.Mvc;
using System.ComponentModel.DataAnnotations;

namespace BackEnd_API.Dtos
{
    public class CreateCustomerDto : Controller
    {
        // para crear tareas -> sin id
        [Required (ErrorMessage = "El nombre es requerido")]
        public string FirstName { get; set; }
        [Required (ErrorMessage = "El Apellido es requerido")]
        public string LastName { get; set; }
        // validar el correo
        [RegularExpression ("^[a-zA-Z0-9_\\-]+@([a-zA-Z]{2,6}$)", ErrorMessage = "El correo es incorrecto")]
        public string Email { get; set; }
        public string Phone { get; set; }
        public string Address { get; set; }
    }
}
