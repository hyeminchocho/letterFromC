#ifdef GL_ES
precision mediump float;
precision mediump int;
#endif

uniform sampler2D texture;
uniform vec3 fontColor;
uniform float lengthRatio;

// varying vec4 vertColor;
varying vec2 vertTexCoord;

void main() {
  float u = (vertTexCoord.s)/lengthRatio;
  vec2 uv = vec2(u, 1.0 - vertTexCoord.t);
  vec4 col = texture2D(texture, uv.st);


  float alpha = smoothstep(0.1, 0.8, (1.0-col.r));
  alpha = alpha * (1.0 - step(1.0, uv.x));
  gl_FragColor = vec4(fontColor, alpha);
  // gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
}
