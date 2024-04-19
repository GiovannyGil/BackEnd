using System.ComponentModel.DataAnnotations;

namespace APITransacciones.Models
{
    public class Cliente
    {
        [Key] // Especifica que ClienteID es la clave primaria
        public int ClienteID { get; set; }
        public required string Nombre { get; set; }
        public required string Email { get; set; }
        public string? Telefono { get; set; }

        // Propiedad para la relacion
        public ICollection<Transaccion> Transacciones { get; set; }

    }
}
