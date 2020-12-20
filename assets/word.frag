#ifdef GL_ES
precision mediump float;
precision mediump int;
#endif

uniform sampler2D texture;
uniform vec3 fontColor;
uniform float lengthRatio;
uniform float minVal;

// varying vec4 vertColor;
varying vec2 vertTexCoord;

void main() {
  float u = (vertTexCoord.s)/lengthRatio;
  vec2 uv = vec2(u, 1.0 - vertTexCoord.t);
  vec4 col = texture2D(texture, uv.st);


  // float alpha = smoothstep(0.07, 0.47, (1.0-col.r));
  float alpha = (1.0-col.r)/minVal*0.9;
  alpha *= smoothstep(0.04, 0.13, (1.0-col.r));
  // float alpha = smoothstep(0.1, 0.8, (1.0-col.r));
  // alpha = 1.0;
  // alpha = alpha * 1.2;
  alpha = alpha * (1.0 - step(1.0, uv.x));
  gl_FragColor = vec4(fontColor, alpha);
  // gl_FragColor = vec4(col.rgb - (1.0 - col.rgb)*0.7, alpha);
  // gl_FragColor = vec4(col.rgb - (1.0 - col.rgb)*0.6, alpha);
  // gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
}
