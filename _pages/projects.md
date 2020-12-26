---
title: "Projects"
layout: gridlay
sitemap: false
permalink: /projects/
---

<style>
.ul {
  padding-left: 10px;
}
.ol {
  padding-left: 10px;
}
.btn{
    margin-bottom:0;
}
.jumbotron{
    padding-bottom:0px;
    padding-top:5px;
    margin-top:10px;
    margin-bottom:10px
}
</style>


## Projects

{% assign yeartest = true %}
{% for project in site.data.projectlist %}
  {% if project.year %}{% else %}
   {% assign yeartest = false %}
  {% endif %}
{% endfor %}

{% if site.group_pub_by_year == true %}{% else %}
### Journal Papers and Proceedings 
{% endif %}

{% for myyear in site.data.years %}

{% assign yeartest = false %}
{% for project in site.data.projectlist %}
  {% if project.year == myyear.year %}
   {% assign yeartest = true %}
  {% endif %}
{% endfor %}

{% if site.group_pub_by_year == true %}
{% if yeartest == true %}
### {{ myyear.year }}
{% endif %}
{% endif %}

{% for project in site.data.projectlist %}
{% if project.year == myyear.year %}


{% assign bibtest = false %}
{% if project.url %}
{% assign bibfile = "/papers/" | append:  project.url  | append: ".txt" %}
{% for file in site.static_files %}
  {% if file.path contains bibfile %}
   {% assign bibtest = true %}
  {% endif %}
{% endfor %}
{% endif %}

<div class="jumbotron">
<div class="row">
<div class="d-none d-md-block col-sm-2">
  {% if project.image %}
   <img  src="{{ site.url }}{{ site.baseurl }}/images/project/{{ project.image }}" width="100%" style="margin-top:10px"/>
  {% endif %}
</div>
<div class="col-md-10 col-sm-12 col-xs-12">
  <b>{{ project.title }}</b><br/>
  {% if project.link %}<a href="{{ project.link }}" target="_blank"><i class="fas fa-link" aria-hidden="true"></i>Project link</a>
   {% endif %}
   {% if project.abstract %} 
   <a data-toggle="collapse" href="#{{project.url}}" aria-expanded="false" aria-controls="{{project.url}}"><i class="fa fa-info-circle" aria-hidden="true"></i>Description</a>
   {% endif %}
  {% if project.abstract %}
<div class="collapse" id="{{project.url}}"><div class="well-collapse">
 {{project.abstract}}
</div></div>
{% endif %} 
{% if project.youtube %} 
   <a data-toggle="collapse" href="#{{project.youtube}}" aria-expanded="false" aria-controls="{{project.youtube}}"><i class="fa fa-file-video-o" aria-hidden="true"></i>VIDEO</a>
   {% endif %}
{% if project.youtube %} 
<div class="collapse" id="{{project.youtube}}"><div class="well-collapse">
 <iframe width="560" height="315" src="https://www.youtube.com/embed/{{project.youtube}}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
 </div></div>
   {% endif %}
</div>
</div>
</div>

{% endif %}
{% endfor %}

{% endfor %}
