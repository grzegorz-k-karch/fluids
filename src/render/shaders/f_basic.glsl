#version 330

uniform sampler2D imageTexture;
uniform vec2 imageScale;

out vec4 out_color;

void main(void) 
{
  out_color = texture(imageTexture, gl_FragCoord.xy*imageScale);
}
