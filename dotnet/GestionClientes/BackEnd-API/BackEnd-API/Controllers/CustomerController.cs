using BackEnd_API.Dtos;
using BackEnd_API.Repositories;
using Microsoft.AspNetCore.Mvc;
using System.Security.Cryptography.X509Certificates;

namespace BackEnd_API.Controllers
{
    [ApiController]
    [Route("api/[controller]")] // traduce a customer en api
    public class CustomerController : Controller
    {
        // inyectar el repositorio/base de datos a la api
        private readonly CustmerDataBaseContext _custmerDataBaseContext;

        public CustomerController(CustmerDataBaseContext custmerDataBaseContext)
        {
            // asiganar el valor inyectado de la BBDD
            _custmerDataBaseContext = custmerDataBaseContext;
        }





        // obtener una sola tarea -> por el id
        //api/customer/{id}
        [HttpGet("{id}")]
        // pasar el tipado y status code
        [ProducesResponseType(StatusCodes.Status200OK, Type = typeof(CustomerDto))]
        [ProducesResponseType(StatusCodes.Status404NotFound)] // no encontrado
        public async Task<IActionResult> GetCustomer(long id)
        {
            CustomerEntity result = await _custmerDataBaseContext.Get(id);
            return new OkObjectResult(result.toDto());
            // si el objeot esta vacio, de vuelve el objeto vacio con el status code 200
        }

        // obtener todas las tareas
        //api/ccustomer/
        [HttpGet]
        // pasar el tipado y status code
        [ProducesResponseType(StatusCodes.Status200OK, Type = typeof(CustomerDto))]
        public async Task<IActionResult> GetCustomers()
        {
            throw new NotImplementedException();
        }

        // eliminar
        [HttpDelete("{id}")]
        // pasar el tipado y status code
        [ProducesResponseType(StatusCodes.Status200OK, Type = typeof(CustomerDto))]
        [ProducesResponseType(StatusCodes.Status404NotFound)] // no encontrado
        public async Task<IActionResult> DeleteCustomer(long id)
        {
            throw new NotImplementedException();
        }

        // agregar la tarea
        [HttpPost]
        // pasar el tipado y status code
        [ProducesResponseType(StatusCodes.Status201Created, Type = typeof(CustomerDto))]
        [ProducesResponseType(StatusCodes.Status404NotFound)] // no encontrado
        public async Task<IActionResult> CreateCustomer(CreateCustomerDto customer)
        {
            CustomerEntity result = await _custmerDataBaseContext.Add(customer);
            return new CreatedResult($"http://localhost:7016/api/customer/{result.Id}", null);
            // si el objeot esta vacio, de vue  lve el objeto vacio con el status code 200
        }

        // actualizar
        [HttpPut]
        // pasar el tipado y status code
        [ProducesResponseType(StatusCodes.Status200OK, Type = typeof(CustomerDto))]
        [ProducesResponseType(StatusCodes.Status404NotFound)] // no encontrado
        public async Task<IActionResult> UpdateCustomer(CustomerDto customer)
        {
            throw new NotImplementedException();
        }
    }
}
