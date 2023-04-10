import openai
import os

def generate_unit_tests(csharp_code, api_key):
    # Set up the OpenAI API client, use input 
    openai.api_key = api_key

    prompt = f"Generate C# unit tests for the following C# code:\n{csharp_code}\n\nUnit tests:\n"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    generated_tests = response.choices[0].text.strip()

    return generated_tests

if __name__ == "__main__":
    csharp_code = '''
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
namespace MVC_BasicTutorials.Controllers
{
    public class StudentController : Controller
    {
        // GET: Student
        public string Index()
        {
                return "This is Index action method of StudentController";
        }
    }
}
    '''

    api_key = input("Enter OpenAI API key:")
    unit_tests = generate_unit_tests(csharp_code, api_key)
    print(unit_tests)