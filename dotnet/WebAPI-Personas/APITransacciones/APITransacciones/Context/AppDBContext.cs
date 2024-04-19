using APITransacciones.Models;
using Microsoft.EntityFrameworkCore;

namespace APITransacciones.Context
{
    public class AppDBContext : DbContext
    {
        public AppDBContext(DbContextOptions<AppDBContext> options) : base(options) { }

        public DbSet<Cliente> Clientes { get; set; }
        public DbSet<Vehiculo> Vehiculos { get; set; }
        public DbSet<Concesionario> Concesionarios { get; set; }
        public DbSet<Transaccion> Transacciones { get; set; }

    }
}
