using System.ComponentModel.DataAnnotations;

namespace APITransacciones.Models
{
      public class Concesionario
    {
        [Key] // Especifica que ConcecionarioID es la clave primaria
        public int ConcecionarioID { get; set; }
        public required string Nombre { get; set; }
        public required string Ciudad { get; set; }
        public string? Direccion { get; set; }

        // Propiedad para la relacion
        public ICollection<Transaccion> Transacciones { get; set; }
    }
}
