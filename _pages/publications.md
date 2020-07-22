---
layout: page
permalink: /publications/
title: Publications
description:
years: [2020, 2019, 2017, 2016, 2015, 2014, 2013, 2012]
---

{% for y in page.years %}
<article class="notepad-index-post post row">

  <h3 datetime="{{ y | date_to_xmlschema }}" class="year" >{{y}}</h3>
   <section class="notepad-post-excerpt">
  {% bibliography -f papers -q @*[year={{y}}]* %}
 </section>    

    </article>
{% endfor %}
