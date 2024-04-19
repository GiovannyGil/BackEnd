using System.ComponentModel.DataAnnotations;

namespace APITransacciones.Models
{
     public class Transaccion
    {
        [Key] // Especifica que TransaccionID es la clave primaria
        public int TransaccionID { get; set; }
        public required int VehiculoID { get; set; }
        public required int ClienteID { get; set; }
        public required int ConcesionarioID { get; set; }
        public required DateTime FechaVenta { get; set; }
        public required int PrecioVenta { get; set; }

    }
}
