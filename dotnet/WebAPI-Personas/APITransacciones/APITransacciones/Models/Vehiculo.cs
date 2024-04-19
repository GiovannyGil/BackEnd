using System.ComponentModel.DataAnnotations;

namespace APITransacciones.Models
{
    public class Vehiculo
    {
        [Key] // Especifica que VehiculoID es la clave primaria
        public int VehiculoID { get; set; }
        public required string Marca { get; set; }
        public required string Modelo { get; set; }
        public required int Año { get; set; }
        public required int Precio { get; set; }

        // Propiedad para la relacion
        public ICollection<Transaccion> Transacciones { get; set; }
    }
}
