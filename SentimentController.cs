using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;

namespace SentimentAPI.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class SentimentController : ControllerBase
    {
        [HttpPost("predict")]
        public IActionResult Predict([FromBody] ReviewRequest request)
        {
            if (string.IsNullOrEmpty(request.Review))
                return BadRequest("Review text is required.");

            try
            {
                // Path to your Python script
                string pythonScript = @"D:\sentimentanalysis\predict_only.py";
                string pythonExe = "python";

                var psi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = $"\"{pythonScript}\" \"{request.Review}\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                string output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();

                return Ok(new { 
                    review = request.Review,
                    sentiment = output.Trim()
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Error: {ex.Message}");
            }
        }
    }

    public class ReviewRequest
    {
        public string Review { get; set; }
    }
}