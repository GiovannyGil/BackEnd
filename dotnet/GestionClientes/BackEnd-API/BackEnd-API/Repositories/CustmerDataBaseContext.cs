using BackEnd_API.Dtos;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.ChangeTracking;

namespace BackEnd_API.Repositories
{
    // conexion a la base de datos
    public class CustmerDataBaseContext : DbContext
    {

        // configuracion
        public CustmerDataBaseContext(DbContextOptions<CustmerDataBaseContext> options): base(options)
        {

        }

        // acceder a la tabal customer de la base de datos
        public DbSet<CustomerEntity> customer { get; set; }

        public async Task<CustomerEntity> Get(long id)
        {
            return await customer.FirstAsync(x => x.Id == id); // 
        }

        public async Task<CustomerEntity> Add(CreateCustomerDto customerDto)
        {
            CustomerEntity entity = new CustomerEntity()
            {
                Id = null,
                Address = customerDto.Address,
                Email = customerDto.Email,
                FirstName = customerDto.FirstName,
                LastName = customerDto.LastName,
                Phone = customerDto.Phone,

            }; // asignar valores

            EntityEntry<CustomerEntity> response = await customer.AddAsync(entity); // generar respuesta
            await SaveChangesAsync(); // guaradar
            return await Get(response.Entity.Id ?? throw new Exception("no se ha podido guardar")); // restornar respuesta o excepcion
        }
    }

    public class CustomerEntity
    {
        public long? Id { get; set; }
        public string FirstName { get; set; }
        public string LastName { get; set; }
        public string Email { get; set; }
        public string Phone { get; set; }
        public string Address { get; set; }


        public CustomerDto toDto()
        {
            return new CustomerDto()
            {
                Address = Address,
                Email = Email,
                FirstName = FirstName,
                LastName = LastName,
                Phone = Phone,
                Id = Id ?? throw new Exception("el ID no puede ser null")
            };
        }
    }
}
